//! FEM-derived structured-grid operators for the hard problem classes a pure
//! stencil Laplacian misses: the **curl-curl** time-harmonic Maxwell operator
//! (complex symmetric indefinite, with a large gradient near-null-space -- the
//! FEM edge-element / EM workload) and the **saddle-point** Stokes/KKT operator
//! (symmetric indefinite, `[A Bᵀ; B -βC]`). Both are assembled directly on a
//! structured grid with finite differences, no external FEM library, following
//! the standard discretizations (curl-curl `∇×∇×E - (ω²ε - iωσ)E`, Nédélec/Yee;
//! Stokes MAC / mixed-FEM `[A Bᵀ; B 0]` with Brezzi-Pitkäranta pressure
//! stabilization `-βC`). References: Jin, *The FEM in Electromagnetics*;
//! Elman-Silvester-Wathen, *Finite Elements and Fast Iterative Solvers* (IFISS);
//! Benzi-Golub-Liesen, *Numerical solution of saddle point problems*.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use num_complex::Complex;

use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;

/// Grid strides for column-major linear indexing of an `ndim`-D grid.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for d in 1..dims.len() {
        s[d] = s[d - 1] * dims[d - 1];
    }
    s
}

/// The discrete divergence row for grid node `k`: central differences over each
/// dimension, `(Du)_k = Σ_a (u[k+e_a,a] - u[k-e_a,a]) / 2`. Returns `(dof, coeff)`
/// entries into a component-major velocity vector (`dof(a,i) = a*n + i`). Missing
/// (boundary) neighbours are dropped.
fn divergence_row(k: usize, dims: &[usize], stride: &[usize], n: usize) -> Vec<(usize, f64)> {
    let ndim = dims.len();
    let mut e = Vec::with_capacity(2 * ndim);
    for a in 0..ndim {
        let c = (k / stride[a]) % dims[a];
        if c + 1 < dims[a] {
            e.push((a * n + (k + stride[a]), 0.5));
        }
        if c > 0 {
            e.push((a * n + (k - stride[a]), -0.5));
        }
    }
    e
}

/// Component-decoupled vector Laplacian (`ndim` blocks) as `(diag, lower-tri)` over
/// the component-major DOF vector, plus a caller-supplied Dirichlet diagonal load.
fn vector_laplacian(
    dims: &[usize],
    stride: &[usize],
    n: usize,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let ndim = dims.len();
    let ndof = ndim * n;
    let mut diag = vec![0.0f64; ndof];
    let mut tri: Vec<(usize, usize, f64)> = Vec::new();
    for a in 0..ndim {
        let off = a * n;
        for i in 0..n {
            for d in 0..ndim {
                let c = (i / stride[d]) % dims[d];
                if c + 1 < dims[d] {
                    let j = i + stride[d];
                    diag[off + i] += 1.0;
                    diag[off + j] += 1.0;
                    tri.push((off + j, off + i, -1.0));
                } else {
                    diag[off + i] += 1.0; // Dirichlet ghost
                }
                if c == 0 {
                    diag[off + i] += 1.0;
                }
            }
        }
    }
    (diag, tri)
}

/// Time-harmonic **curl-curl** Maxwell operator on a structured grid:
/// `A = (∇×∇×) - ω²M + iωσM`, discretized via the identity `∇×∇× = L - grad·div`
/// (`L` the component-wise vector Laplacian, `grad·div = DᵀD` from the discrete
/// divergence `D`). `M` is the lumped mass (identity). The `-ω²M` shift makes the
/// operator **indefinite**; the `iωσM` conductivity term makes it **complex
/// symmetric** (`A = Aᵀ`, not Hermitian); the curl-curl part is singular on
/// discrete gradients, so the system has the gradient near-null-space that makes
/// edge-element EM problems hard for iterative solvers and a stress test for a
/// direct one. `dims` is `[nx, ny]` (2 components) or `[nx, ny, nz]` (3).
pub fn curl_curl(dims: &[usize], omega: f64, sigma: f64) -> CscMatrix<Complex<f64>> {
    let ndim = dims.len();
    let n: usize = dims.iter().product();
    let ndof = ndim * n;
    let stride = strides(dims);

    // Off-diagonal accumulator keyed by (hi, lo) with hi > lo (lower triangle).
    let mut off: HashMap<(usize, usize), f64> = HashMap::new();
    let mut diag = vec![0.0f64; ndof];

    // + L (vector Laplacian).
    let (l_diag, l_tri) = vector_laplacian(dims, &stride, n);
    for (i, d) in l_diag.iter().enumerate() {
        diag[i] += *d;
    }
    for (r, c, w) in l_tri {
        *off.entry((r, c)).or_insert(0.0) += w;
    }
    // - DᵀD (grad-div): accumulate the outer product of each divergence row.
    for k in 0..n {
        let e = divergence_row(k, dims, &stride, n);
        for a in 0..e.len() {
            let (p, cp) = e[a];
            diag[p] -= cp * cp;
            for b in (a + 1)..e.len() {
                let (q, cq) = e[b];
                let (hi, lo) = if p > q { (p, q) } else { (q, p) };
                *off.entry((hi, lo)).or_insert(0.0) -= cp * cq;
            }
        }
    }

    // Complex mass shift on the diagonal: -ω²M + iωσM (M = I).
    let shift = Complex::new(-omega * omega, omega * sigma);
    let mut rows = Vec::with_capacity(ndof + off.len());
    let mut cols = Vec::with_capacity(ndof + off.len());
    let mut vals = Vec::with_capacity(ndof + off.len());
    for i in 0..ndof {
        rows.push(i);
        cols.push(i);
        vals.push(Complex::new(diag[i], 0.0) + shift);
    }
    for ((r, c), w) in off {
        if w != 0.0 {
            rows.push(r);
            cols.push(c);
            vals.push(Complex::new(w, 0.0));
        }
    }
    super::build_sym(ndof, &rows, &cols, &vals)
}

/// **Saddle-point** Stokes/KKT operator `[A Bᵀ; B -βC]` on a structured grid
/// (symmetric indefinite), generic over the scalar type. `A` is the velocity
/// vector Laplacian (SPD), `B` the discrete divergence coupling velocity to
/// pressure, and `-βC` a Brezzi-Pitkäranta pressure-Laplacian stabilization
/// (`β>0`) that regularizes the otherwise-singular zero `(2,2)` block on a
/// collocated grid. DOF layout: velocity `a*n+i` (`a<ndim`), then pressure
/// `ndim*n + i`. The zero/negative pressure block makes the system indefinite --
/// the KKT/constrained-optimization class distinct from an SPD or a shifted
/// Helmholtz operator.
pub fn saddle_point<T: Scalar>(dims: &[usize], beta: f64) -> CscMatrix<T> {
    let ndim = dims.len();
    let n: usize = dims.iter().product();
    let stride = strides(dims);
    let nu = ndim * n; // velocity DOFs
    let ndof = nu + n; // + pressure DOFs
    let pdof = |i: usize| nu + i;

    let mut diag = vec![0.0f64; ndof];
    let mut tri: Vec<(usize, usize, f64)> = Vec::new();

    // A: velocity vector Laplacian (top-left, SPD).
    let (l_diag, l_tri) = vector_laplacian(dims, &stride, n);
    for (i, d) in l_diag.iter().enumerate() {
        diag[i] += *d;
    }
    tri.extend(l_tri);

    // B / Bᵀ: divergence coupling. Row = pressure dof (always > velocity dofs), so
    // (pdof(k), vel_dof) is lower-triangle; symmetry supplies Bᵀ.
    for k in 0..n {
        for (v, coeff) in divergence_row(k, dims, &stride, n) {
            tri.push((pdof(k), v, coeff));
        }
    }

    // -βC: pressure Laplacian stabilization (bottom-right, negative definite).
    let (c_diag, c_tri) = vector_laplacian(dims, &stride, n); // ndim blocks; use block 0 (scalar)
    for i in 0..n {
        diag[pdof(i)] -= beta * c_diag[i];
    }
    for (r, c, w) in c_tri {
        if r < n && c < n {
            // block-0 (scalar) entries only
            tri.push((pdof(r), pdof(c), -beta * w));
        }
    }

    let mut rows = Vec::with_capacity(ndof + tri.len());
    let mut cols = Vec::with_capacity(ndof + tri.len());
    let mut vals = Vec::with_capacity(ndof + tri.len());
    for i in 0..ndof {
        rows.push(i);
        cols.push(i);
        vals.push(T::from_real(diag[i]));
    }
    for (r, c, w) in tri {
        rows.push(r);
        cols.push(c);
        vals.push(T::from_real(w));
    }
    super::build_sym(ndof, &rows, &cols, &vals)
}

/// Advection field for [`convection_diffusion`].
#[derive(Clone, Copy, Debug)]
pub enum Flow {
    /// Constant unit velocity along the grid diagonal (equal on every axis) --- a
    /// uniform wind, unsymmetric but spatially constant.
    Diagonal,
    /// Recirculating vortex in the first two axes, `b = (-(y-½), (x-½))`, zero on
    /// any third axis. The classic double-glazing / recirculating-flow benchmark:
    /// the velocity reverses across the domain, so up-/down-wind coupling flips
    /// sign, a richer unsymmetric structure than a constant wind.
    Rotating,
}

/// Convection--diffusion operator `-ε∇²u + b·∇u = f` on a structured `dims` grid
/// with interior unknowns and homogeneous Dirichlet boundaries, finite-differenced
/// --- the canonical source of **unsymmetric** sparse matrices (the LU path's
/// workload). The first-derivative advection term `b·∇u` breaks symmetry; the
/// diffusion term is the usual 5/7-point Laplacian scaled by `eps`.
///
/// * `upwind=false` (central differences) is the sharp, oscillation-prone form
///   that stresses pivoting at small `eps` (high grid-Péclet `|b|h/ε`);
/// * `upwind=true` is the first-order upwind form, a diagonally dominant M-matrix.
///
/// Sweeping `eps` moves the operator from diffusion-dominated (nearly symmetric,
/// small Péclet) to advection-dominated (strongly unsymmetric), and [`Flow`]
/// varies the advection field --- together they span the unsymmetric distribution
/// on a structured grid without an external FEM library.
pub fn convection_diffusion<T: Scalar>(
    dims: &[usize],
    eps: f64,
    flow: Flow,
    upwind: bool,
) -> GeneralCsc<T> {
    let d = dims.len();
    let s = strides(dims);
    let n: usize = dims.iter().product();
    let h: Vec<f64> = dims.iter().map(|&m| 1.0 / (m as f64 + 1.0)).collect();
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..n {
        // Multi-index and physical coordinates in (0,1)^d of interior node p.
        let ci: Vec<usize> = (0..d).map(|k| (p / s[k]) % dims[k]).collect();
        let x: Vec<f64> = (0..d).map(|k| (ci[k] as f64 + 1.0) * h[k]).collect();
        // Advection velocity b(x).
        let b: Vec<f64> = match flow {
            Flow::Diagonal => vec![1.0 / (d as f64).sqrt(); d],
            Flow::Rotating => (0..d)
                .map(|k| match k {
                    0 => -(x[1] - 0.5),
                    1 => x[0] - 0.5,
                    _ => 0.0,
                })
                .collect(),
        };
        let mut diag = 0.0;
        for k in 0..d {
            let diff = eps / (h[k] * h[k]);
            diag += 2.0 * diff;
            let has_lo = ci[k] > 0;
            let has_hi = ci[k] + 1 < dims[k];
            let lo = p.wrapping_sub(s[k]);
            let hi = p + s[k];
            // Diffusion off-diagonals (Dirichlet: a boundary neighbour is u=0, so
            // its off-diagonal entry is simply omitted; the diagonal term stays).
            if has_lo {
                rows.push(p);
                cols.push(lo);
                vals.push(T::from_real(-diff));
            }
            if has_hi {
                rows.push(p);
                cols.push(hi);
                vals.push(T::from_real(-diff));
            }
            // Advection: central `b_k(u_hi - u_lo)/(2h)` or first-order upwind.
            if upwind {
                let a = b[k] / h[k];
                if b[k] >= 0.0 {
                    diag += a;
                    if has_lo {
                        rows.push(p);
                        cols.push(lo);
                        vals.push(T::from_real(-a));
                    }
                } else {
                    diag -= a; // |b_k|/h
                    if has_hi {
                        rows.push(p);
                        cols.push(hi);
                        vals.push(T::from_real(a));
                    }
                }
            } else {
                let a = b[k] / (2.0 * h[k]);
                if has_hi {
                    rows.push(p);
                    cols.push(hi);
                    vals.push(T::from_real(a));
                }
                if has_lo {
                    rows.push(p);
                    cols.push(lo);
                    vals.push(T::from_real(-a));
                }
            }
        }
        rows.push(p);
        cols.push(p);
        vals.push(T::from_real(diag));
    }
    super::build_gen(n, &rows, &cols, &vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LdltSymbolic, SolverSettings, ZeroPivotAction};

    fn cube(k: usize) -> [usize; 3] {
        [k, k, k]
    }

    #[test]
    fn curl_curl_is_complex_symmetric_indefinite_and_factors() {
        let a = curl_curl(&cube(8), 2.0, 0.3);
        // Genuinely complex (the conductivity term).
        let max_im = a.values.iter().map(|v| v.im.abs()).fold(0.0, f64::max);
        assert!(max_im > 0.0, "curl-curl carries complex values");
        // Factors via the preconditioner path (indefinite / near-singular curl-curl
        // needs perturbation), then refines to a small residual.
        let sym = LdltSymbolic::analyze(&a).unwrap();
        let opts = SolverSettings::default()
            .with_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 });
        let solver = sym.factor(&a, &opts).unwrap();
        let n = a.n;
        let b: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i % 5) as f64 - 2.0, 0.5))
            .collect();
        let x = solver.solve_refined(&a, &b, 40).unwrap();
        let mut ax = vec![Complex::new(0.0, 0.0); n];
        a.symv(&x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max)
            / b.iter().map(|v| v.norm()).fold(0.0, f64::max).max(1e-30);
        assert!(res < 1e-6, "curl-curl refined residual {res}");
    }

    #[test]
    fn saddle_point_is_symmetric_indefinite_and_factors() {
        let a = saddle_point::<f64>(&[10usize, 10], 0.1);
        // The pressure block makes it indefinite: a plain Cholesky-style exact
        // factor would hit a non-positive pivot; Bunch-Kaufman handles it.
        let sym = LdltSymbolic::analyze(&a).unwrap();
        let opts = SolverSettings::default()
            .with_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-12 });
        let solver = sym.factor(&a, &opts).unwrap();
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let x = solver.solve_refined(&a, &b, 40).unwrap();
        let mut ax = vec![0.0; n];
        a.symv(&x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).abs()).fold(0.0, f64::max)
            / b.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
        assert!(res < 1e-6, "saddle-point refined residual {res}");
    }

    #[test]
    fn convection_diffusion_is_unsymmetric_and_factors() {
        use crate::LuSymbolic;
        // Advection-dominated (small eps), recirculating flow, upwind: a well-posed
        // unsymmetric M-matrix that must factor through the LU path.
        let a = convection_diffusion::<f64>(&[20usize, 20], 0.01, Flow::Rotating, true);
        // Unsymmetric by construction: find a transposed entry pair A[i,j] != A[j,i].
        let get = |i: usize, j: usize| -> f64 {
            let (s, e) = (a.col_ptr[j], a.col_ptr[j + 1]);
            a.row_idx[s..e]
                .iter()
                .position(|&r| r == i)
                .map(|k| a.values[s + k])
                .unwrap_or(0.0)
        };
        let mut asymmetric = false;
        for j in 0..a.n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[k];
                if i != j && (a.values[k] - get(j, i)).abs() > 1e-12 {
                    asymmetric = true;
                }
            }
        }
        assert!(asymmetric, "convection-diffusion is unsymmetric");
        // Factors and solves through the unsymmetric LU path.
        let sym = LuSymbolic::analyze(&a).unwrap();
        let solver = sym.factor(&a, &SolverSettings::default()).unwrap();
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let x = solver.solve(&b).unwrap();
        let mut ax = vec![0.0; n];
        a.matvec(&x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).abs()).fold(0.0, f64::max)
            / b.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
        assert!(res < 1e-8, "convection-diffusion LU residual {res}");
    }
}
