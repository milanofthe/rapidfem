use super::*;
use crate::error::RslabError;
use num_complex::Complex;
type C = Complex<f64>;

/// Warm start through the closure adapters: seeding `gmres_block` with the
/// exact solution must converge immediately (0 iterations past the residual
/// check), and with a perturbed seed in strictly fewer iterations than cold.
#[test]
fn closure_warm_start_converges_immediately() {
    let n = 40;
    let s = 2;
    // Diagonally dominant complex system, closure matvec.
    let a = |i: usize, j: usize| -> C {
        if i == j {
            C::new(4.0 + i as f64 * 0.01, 0.5)
        } else {
            C::new(0.3 / (1.0 + (i as f64 - j as f64).abs()), -0.1)
        }
    };
    let apply = |x: &[C], y: &mut [C], cols: usize| {
        for c in 0..cols {
            for i in 0..n {
                let mut acc = C::new(0.0, 0.0);
                for j in 0..n {
                    acc += a(i, j) * x[j + c * n];
                }
                y[i + c * n] = acc;
            }
        }
    };
    let op = FnOperator::new(n, apply);
    let ident = FnPreconditioner::new(
        |r: &[C], z: &mut [C], _s: usize| -> Result<(), RslabError> {
            z.copy_from_slice(r);
            Ok(())
        },
    );
    let xs: Vec<C> = (0..n * s)
        .map(|k| C::new((k as f64 * 0.37).sin(), (k as f64 * 0.11).cos()))
        .collect();
    let mut b = vec![C::new(0.0, 0.0); n * s];
    op.apply_block(&xs, &mut b, s);

    let cold = gmres_block(
        &op,
        &b,
        s,
        &ident,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(500)
            .with_restart(30),
        None,
        None,
    )
    .expect("cold");
    assert!(cold.iters > 3, "cold solve trivial: {}", cold.iters);

    let warm = gmres_block(
        &op,
        &b,
        s,
        &ident,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(500)
            .with_restart(30),
        Some(&xs),
        None,
    )
    .expect("warm");
    assert_eq!(warm.iters, 0, "exact seed must converge immediately");
    for k in 0..n * s {
        assert!((warm.x[k] - xs[k]).norm() < 1e-8);
    }

    // Perturbed seed: strictly fewer iterations than cold.
    let near: Vec<C> = xs.iter().map(|v| v * C::new(1.001, 0.0)).collect();
    let warm2 = gmres_block(
        &op,
        &b,
        s,
        &ident,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(500)
            .with_restart(30),
        Some(&near),
        None,
    )
    .expect("warm2");
    assert!(
        warm2.iters < cold.iters,
        "near seed not faster: {} vs {}",
        warm2.iters,
        cold.iters
    );
}
