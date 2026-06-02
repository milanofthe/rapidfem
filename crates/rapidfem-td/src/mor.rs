//! Model-order reduction by Krylov-subspace projection.
//!
//! An `r`-step Arnoldi process on the (matrix-free) operator `A` from a start
//! vector builds an orthonormal basis `V` (`n×r`) and the projected operator
//! `Â = VᵀAV` (`r×r`, the Arnoldi Hessenberg). The reduced model
//! `dŷ/dt = Â·ŷ`, `y ≈ V·ŷ` captures the dynamics in the Krylov subspace at a
//! fraction of the cost — the same machinery that powers the exponential
//! propagator, here kept as a standalone object.

use crate::constants::ARNOLDI_BREAKDOWN;
use crate::propagator::expm;

/// A reduced-order model: `A ≈ V·Â·Vᵀ` with `Â` small and dense.
pub struct ReducedModel {
    /// Reduced dimension.
    pub r: usize,
    /// Full dimension.
    pub n: usize,
    /// Orthonormal basis — `r` vectors of length `n`.
    basis: Vec<Vec<f64>>,
    /// Reduced operator `Â = VᵀAV`, `r×r` row-major.
    pub a_hat: Vec<f64>,
}

impl ReducedModel {
    /// Build via `r`-step Arnoldi on `matvec` from the vector `start`.
    /// Stops early on a lucky breakdown.
    pub fn build<F>(matvec: F, start: &[f64], r: usize) -> Self
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = start.len();
        let beta = norm2(start);
        assert!(beta > 0.0, "start vector must be nonzero");
        let mut basis: Vec<Vec<f64>> =
            vec![start.iter().map(|x| x / beta).collect()];
        let mut h = vec![0.0; r * r];
        let mut dim = r;

        for j in 0..r {
            let mut w = matvec(&basis[j]);
            for i in 0..=j {
                let hij = dot(&w, &basis[i]);
                h[i * r + j] = hij;
                for k in 0..n {
                    w[k] -= hij * basis[i][k];
                }
            }
            let hn = norm2(&w);
            if hn < ARNOLDI_BREAKDOWN {
                dim = j + 1;
                break;
            }
            if j + 1 < r {
                h[(j + 1) * r + j] = hn;
                basis.push(w.iter().map(|x| x / hn).collect());
            }
        }

        // Extract the dim×dim leading block.
        let mut a_hat = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                a_hat[i * dim + j] = h[i * r + j];
            }
        }
        ReducedModel { r: dim, n, basis, a_hat }
    }

    /// Project a full state into the reduced space — `ŷ = Vᵀ·y`.
    pub fn project(&self, y: &[f64]) -> Vec<f64> {
        self.basis.iter().map(|v| dot(v, y)).collect()
    }

    /// Lift a reduced state back to the full space — `y = V·ŷ`.
    pub fn lift(&self, yhat: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n];
        for (c, v) in yhat.iter().zip(&self.basis) {
            for k in 0..self.n {
                y[k] += c * v[k];
            }
        }
        y
    }

    /// Propagate a full state by `t` through the reduced model —
    /// `V·exp(t·Â)·Vᵀ·y₀`.
    pub fn propagate(&self, y0: &[f64], t: f64) -> Vec<f64> {
        let yhat = self.project(y0);
        let th: Vec<f64> = self.a_hat.iter().map(|x| x * t).collect();
        let e = expm(&th, self.r);
        let mut yt = vec![0.0; self.r];
        for i in 0..self.r {
            for j in 0..self.r {
                yt[i] += e[i * self.r + j] * yhat[j];
            }
        }
        self.lift(&yt)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_gen::structured_box;
    use crate::propagator::expm;
    use crate::rhs::MaxwellOperator;

    #[test]
    fn reduced_model_reproduces_full_propagation() {
        // A Krylov ROM of dimension r ≪ N reproduces the full propagation
        // exp(tA)·y₀ of the DG Maxwell operator.
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let start: Vec<f64> =
            (0..n).map(|i| (0.5 + i as f64 * 0.021).sin()).collect();

        let rom = ReducedModel::build(|x| op.apply(x), &start, 70);
        assert!(rom.r <= 70 && rom.r < n, "r={} n={n}", rom.r);

        let t = 0.05;
        // Dense full reference exp(tA)·start.
        let a = op.assemble_dense();
        let ta: Vec<f64> = a.iter().map(|x| x * t).collect();
        let e = expm(&ta, n);
        let mut want = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                want[i] += e[i * n + j] * start[j];
            }
        }

        let got = rom.propagate(&start, t);
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = want.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!(
            err < 1e-6 * scale,
            "ROM r={} vs full: rel.err {}",
            rom.r,
            err / scale
        );
    }

    #[test]
    fn project_then_lift_is_identity_on_the_subspace() {
        // V is orthonormal ⇒ lift(project(·)) is the orthogonal projector;
        // applied to a basis vector it returns it unchanged.
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let start: Vec<f64> = (0..n).map(|i| (i as f64 * 0.03).cos()).collect();
        let rom = ReducedModel::build(|x| op.apply(x), &start, 40);

        let round = rom.lift(&rom.project(&start));
        let err: f64 = round
            .iter()
            .zip(&start)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = start.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-10 * scale, "projector off by {}", err / scale);
    }
}
