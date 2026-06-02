use num_complex::Complex64 as C64;
use rapidfem_fd::pardiso::{PardisoSolver, build_upper_csr};

#[test]
fn test_pardiso_2x2() {
    let mut solver = match PardisoSolver::try_new() {
        Some(s) => s,
        // Stable libtest has no skip outcome, so a missing backend would
        // otherwise report as a green pass. Print a SKIP marker the report
        // bridge reclassifies to "skipped" when the MKL runtime is absent.
        None => { eprintln!("SKIP: PARDISO runtime not available (MKL not loaded)"); return; }
    };

    // A = [[2+1j, 1], [1, 3+2j]] — same as Python test
    // Full COO (both triangles):
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let vals = vec![
        C64::new(2.0, 1.0), C64::new(1.0, 0.0),
        C64::new(1.0, 0.0), C64::new(3.0, 2.0),
    ];

    let (ia, ja, a) = build_upper_csr(2, &rows, &cols, &vals);
    eprintln!("ia = {:?}", ia);
    eprintln!("ja = {:?}", ja);
    eprintln!("a = {:?}", a);

    // Expected upper triangle CSR: ia=[0,2,3], ja=[0,1,1], a=[(2+1j),(1+0j),(3+2j)]
    assert_eq!(ia, vec![0, 2, 3]);
    assert_eq!(ja, vec![0, 1, 1]);
    assert_eq!(a.len(), 3);

    solver.analyze_and_factorize(2, &ia, &ja, &a).expect("PARDISO failed");

    let rhs = vec![C64::new(1.0, 0.0), C64::new(2.0, 0.0)];
    let x = solver.solve(2, &ia, &ja, &a, &rhs).expect("PARDISO solve failed");

    eprintln!("x = {:?}", x);

    // Verify Ax = rhs
    let ax0 = C64::new(2.0, 1.0) * x[0] + C64::new(1.0, 0.0) * x[1];
    let ax1 = C64::new(1.0, 0.0) * x[0] + C64::new(3.0, 2.0) * x[1];
    let err = ((ax0 - rhs[0]).norm() + (ax1 - rhs[1]).norm());
    eprintln!("residual = {:.2e}", err);
    assert!(err < 1e-10, "PARDISO 2x2 residual too large: {}", err);
    eprintln!("PARDISO 2x2: PASS");
}
