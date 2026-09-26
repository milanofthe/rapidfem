/// The MC64 transversal keeps the diagonal-preference pivoting on the
/// diagonal: on a row-scrambled, badly scaled grid the numeric fill
/// equals the symbolic estimate, while the structural transversal
/// pivots off the diagonal and fills.
#[test]
fn klu_matching_keeps_fill_at_the_symbolic_estimate() {
    let m = 40usize;
    let n = m * m;
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for j in 0..n {
        let (x, y) = (j % m, j / m);
        let rs = |i: usize| 10f64.powi(((i * 7919) % 13) as i32 - 6);
        let mut push = |i: usize, v: f64| cols[j].push(((i + 17) % n, v * rs(i)));
        push(j, 4.0);
        if x > 0 {
            push(j - 1, -1.4);
        }
        if x + 1 < m {
            push(j + 1, -0.6);
        }
        if y > 0 {
            push(j - m, -1.0);
        }
        if y + 1 < m {
            push(j + m, -1.0);
        }
    }
    let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
    for c in &mut cols {
        c.sort_by_key(|e| e.0);
        for &(r, v) in c.iter() {
            row_idx.push(r);
            values.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    let a = GeneralCsc {
        n,
        col_ptr,
        row_idx,
        values,
    };
    let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
    let with = KluSettings::default();
    let sym = KluSymbolic::analyze(&a, &with).unwrap();
    let s = sym.factor(&a, &with).unwrap();
    assert_eq!(
        s.factor_nnz(),
        sym.symbolic_factor_nnz(),
        "no pivot-induced fill"
    );
    let x = s.solve(&b).unwrap();
    let mut r = b.clone();
    let mut d: Vec<f64> = b.iter().map(|v| v.abs()).collect();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            r[a.row_idx[k]] -= a.values[k] * x[j];
            d[a.row_idx[k]] += a.values[k].abs() * x[j].abs();
        }
    }
    let omega = r
        .iter()
        .zip(&d)
        .map(|(ri, di)| ri.abs() / di)
        .fold(0.0, f64::max);
    assert!(omega < 1e-13, "backward error {omega}");
    let without = KluSettings::default().with_matching(false);
    let s0 = KluSolver::factor(&a, &without).unwrap();
    assert!(
        s0.factor_nnz() > s.factor_nnz(),
        "structural transversal fills more"
    );
}
use super::*;
use crate::numeric::lu::LuSolver;
use crate::numeric::settings::SolverSettings;
use num_complex::Complex;

fn resid<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
    let mut ax = vec![T::zero(); a.n];
    a.matvec(x, &mut ax);
    let num = b
        .iter()
        .zip(&ax)
        .map(|(&bi, &axi)| (bi - axi).magnitude())
        .fold(0.0, f64::max);
    let den = b.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
    num / den.max(1e-300)
}

/// Deterministic xorshift for value generation (no rand dependency).
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// Circuit-shaped test matrix: sparse, unsymmetric, diagonally weighted,
/// structurally nonsingular, with genuinely reducible structure (a
/// one-directional bridge between two internally coupled halves).
fn circuit_like(n: usize, seed: u64) -> GeneralCsc<f64> {
    let mut rng = Rng(seed | 1);
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let half = n / 2;
    for j in 0..n {
        r.push(j);
        c.push(j);
        v.push(4.0 + rng.next_f64());
        // couplings within the same half only (keeps two SCC groups)
        let base = if j < half { 0 } else { half };
        let span = if j < half { half } else { n - half };
        for t in 1..=3usize {
            let i = base + (j - base + t * 7 + 1) % span;
            if i != j {
                r.push(i);
                c.push(j);
                v.push(rng.next_f64());
            }
        }
    }
    // one-directional bridge: second half feeds the first (rows in the
    // first half, columns in the second) -> reducible, never a single SCC
    for k in 0..4usize {
        r.push(k * 3 % half);
        c.push(half + (k * 5) % (n - half));
        v.push(0.5 + rng.next_f64().abs());
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

/// Off-diagonal pivots must not cascade: when a tiny diagonal forces an
/// off-diagonal pivot, the displaced diagonal row is reassigned to the
/// column whose diagonal row was stolen (SuiteSparse KLU's repair), so
/// later columns keep their diagonal preference and the fill stays near
/// the symbolic diagonal-pivoting prediction instead of degenerating
/// toward partial-pivoting fill (the scircuit/rajat15 2x fill blow-up).
#[test]
fn klu_offdiagonal_pivot_reassigns_diagonal() {
    // scircuit's mechanism in miniature: power-net rows (uniform
    // conductances -> every entry is that row's max, so row-max scaling
    // turns them into permanent large pivot candidates in every column
    // they touch) plus a few tiny diagonals that force the first steal.
    let n = 900;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        for t in [1usize, 2] {
            r.push((j + t) % n);
            c.push(j);
            v.push(1.0);
        }
        let tiny = j % 21 == 5; // j = 2 mod 3: never on a power-net column
        r.push(j);
        c.push(j);
        v.push(if tiny { 1e-9 } else { 4.0 });
        if tiny {
            // the steal target: dominant candidate in the tiny column...
            r.push(j + 1);
            c.push(j);
            v.push(10.0);
            // ...whose own column keeps the displaced row available, but
            // small enough that plain partial pivoting would not pick it
            r.push(j);
            c.push(j + 1);
            v.push(0.3);
        }
    }
    // power nets: rows touching every 3rd column with uniform values
    for k in 0..6usize {
        let p = 100 + 130 * k;
        for j in (0..n).step_by(3) {
            if j != p && j + 1 != p && j + 2 != p {
                r.push(p);
                c.push(j);
                v.push(2.0);
            }
        }
    }
    let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
    let sym = KluSymbolic::analyze(&a, &KluSettings::default()).unwrap();
    let f = sym.factor(&a, &KluSettings::default()).unwrap();
    assert!(
        (f.factor_nnz() as f64) < 1.5 * sym.symbolic_factor_nnz() as f64,
        "off-diagonal pivots cascaded: fill {} vs symbolic {}",
        f.factor_nnz(),
        sym.symbolic_factor_nnz()
    );
    let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
    let x = f.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
}

#[test]
fn klu_solves_circuit_like_and_matches_lu() {
    let a = circuit_like(200, 42);
    let b: Vec<f64> = (0..200).map(|i| (i % 11) as f64 - 5.0).collect();
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert!(s.n_blocks() >= 2, "bridge structure must be reducible");
    let x = s.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-12, "residual {}", resid(&a, &x, &b));
    // cross-check against the supernodal LU
    let f = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let xr = f.solve(&b).unwrap();
    let diff = x
        .iter()
        .zip(&xr)
        .map(|(&p, &q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(diff < 1e-9, "klu vs lu differ by {diff}");
}

#[test]
fn klu_complex_small_diagonal_pivots() {
    // Small diagonal, large off-diagonals: threshold pivoting must
    // abandon the diagonal and still solve accurately (same layout as
    // the supernodal LU pivoting test).
    let c = |re, im| Complex::new(re, im);
    let m = 6;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(0.3, 0.05));
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(c(2.0, 0.3));
                rr.push(q);
                cc.push(p);
                vv.push(c(1.5, -0.2));
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(c(1.8, 0.1));
                rr.push(q);
                cc.push(p);
                vv.push(c(2.2, 0.4));
            }
        }
    }
    let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let x = s.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-12, "residual {}", resid(&a, &x, &b));
}

#[test]
fn klu_lower_triangular_needs_no_fill() {
    // Lower bidiagonal: BTF flips it upper triangular; every block is a
    // singleton, so the factor stores no L/U entries at all.
    let n = 50;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(2.0 + (i % 3) as f64);
        if i + 1 < n {
            r.push(i + 1);
            c.push(i);
            v.push(-1.0);
        }
    }
    let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert_eq!(s.n_blocks(), n);
    // factor_nnz = n diagonal entries + (n-1) off-block entries, zero fill
    assert_eq!(s.factor_nnz(), 2 * n - 1);
    let b: Vec<f64> = (0..n).map(|i| i as f64 - 7.0).collect();
    let x = s.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-14);
}

#[test]
fn klu_structurally_singular_detected() {
    // Column 2 shares its only row pattern with column 0 -> no complete
    // matching regardless of values.
    let a = GeneralCsc::<f64>::from_triplets(3, &[0, 1, 0], &[0, 1, 2], &[1.0, 1.0, 5.0]).unwrap();
    match KluSymbolic::analyze(&a, &KluSettings::default()) {
        Err(RslabError::StructurallySingular) => {}
        other => panic!("expected StructurallySingular, got {other:?}"),
    }
}

#[test]
fn klu_numerically_singular_detected() {
    // Structurally fine 2x2 block, but rank 1 numerically: the second
    // pivot must come up exactly zero.
    let a =
        GeneralCsc::<f64>::from_triplets(2, &[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 2.0, 2.0, 4.0])
            .unwrap();
    match KluSolver::factor(&a, &KluSettings::default()) {
        Err(RslabError::SingularBasis { .. }) => {}
        other => panic!("expected SingularBasis, got {other:?}"),
    }
}

#[test]
fn klu_factor_is_bit_deterministic() {
    let a = circuit_like(150, 7);
    let s1 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let s2 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert_eq!(s1.factors.l_val, s2.factors.l_val);
    assert_eq!(s1.factors.u_val, s2.factors.u_val);
    assert_eq!(s1.factors.udiag, s2.factors.udiag);
    assert_eq!(s1.factors.row_perm, s2.factors.row_perm);
    let b: Vec<f64> = (0..150).map(|i| (i % 13) as f64).collect();
    assert_eq!(s1.solve(&b).unwrap(), s2.solve(&b).unwrap());
}

#[test]
fn klu_refactor_replays_and_matches_fresh_factor() {
    let a = circuit_like(150, 99);
    let mut s = KluSolver::factor(&a, &KluSettings::default()).unwrap();

    // Same values: the replay must reproduce the factor bit-identically.
    let (lv, uv, dv) = (
        s.factors.l_val.clone(),
        s.factors.u_val.clone(),
        s.factors.udiag.clone(),
    );
    s.refactor(&a).unwrap();
    assert_eq!(s.factors.l_val, lv);
    assert_eq!(s.factors.u_val, uv);
    assert_eq!(s.factors.udiag, dv);

    // New values, same pattern: the refactored solve must be accurate.
    let a2 = GeneralCsc::from_triplets(
        a.n,
        &{
            let mut rows = Vec::new();
            for j in 0..a.n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    rows.push(a.row_idx[k]);
                }
            }
            rows
        },
        &{
            let mut cols = Vec::new();
            for j in 0..a.n {
                for _ in a.col_ptr[j]..a.col_ptr[j + 1] {
                    cols.push(j);
                }
            }
            cols
        },
        &a.values
            .iter()
            .enumerate()
            .map(|(k, &v)| v * (1.0 + 0.01 * ((k % 17) as f64)))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    s.refactor(&a2).unwrap();
    let b: Vec<f64> = (0..a.n).map(|i| (i % 9) as f64 - 4.0).collect();
    let x = s.solve(&b).unwrap();
    assert!(
        resid(&a2, &x, &b) < 1e-11,
        "refactor residual {}",
        resid(&a2, &x, &b)
    );
}

/// 2D convection-diffusion 5-point grid: one large irreducible block with
/// wide elimination-DAG wavefronts - the level-schedule target shape.
fn grid_cd(m: usize) -> (GeneralCsc<f64>, Vec<f64>) {
    let n = m * m;
    let idx = |i: usize, j: usize| i * m + j;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..m {
        for j in 0..m {
            let p = idx(i, j);
            r.push(p);
            c.push(p);
            v.push(4.0 + 0.01 * (p % 7) as f64);
            let mut off = |q: usize, w: f64| {
                r.push(q);
                c.push(p);
                v.push(w);
            };
            if i > 0 {
                off(idx(i - 1, j), -1.2);
            }
            if i + 1 < m {
                off(idx(i + 1, j), -0.8);
            }
            if j > 0 {
                off(idx(i, j - 1), -1.1);
            }
            if j + 1 < m {
                off(idx(i, j + 1), -0.9);
            }
        }
    }
    let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    (a, b)
}

#[test]
fn klu_pipelined_refactor_is_bit_identical_to_sequential() {
    // One irreducible 1024-column block: the level schedule must engage
    // (parallel != Off) and its refactor replay must be bit-identical to
    // the strictly sequential one - same values, not just same residual.
    let (a, b) = grid_cd(32);
    let seq = KluSettings::default().with_parallel(KluParallel::Off);
    let par = KluSettings::default().with_parallel(KluParallel::On);

    let mut s_seq = KluSolver::factor(&a, &seq).unwrap();
    let mut s_par = KluSolver::factor(&a, &par).unwrap();
    assert!(
        s_seq.factors.pipelined.is_empty(),
        "Off must not admit pipeline blocks"
    );
    // The 1024-column test block is far below the work gate; force the
    // admission so the pipelined executor itself is exercised (the gates
    // only decide when it pays, not whether it is correct).
    s_par.factors.pipelined = vec![(0, 4)];

    // Fresh values on the frozen pattern, refactor both ways.
    let mut a2 = a.clone();
    for (k, v) in a2.values.iter_mut().enumerate() {
        *v += 1e-3 * ((k % 11) as f64 - 5.0);
    }
    s_seq.refactor(&a2).unwrap();
    s_par.refactor(&a2).unwrap();
    assert_eq!(s_seq.factors.l_val, s_par.factors.l_val, "L values differ");
    assert_eq!(s_seq.factors.u_val, s_par.factors.u_val, "U values differ");
    assert_eq!(s_seq.factors.udiag, s_par.factors.udiag, "diag differs");

    let x = s_par.solve(&b).unwrap();
    assert!(
        resid(&a2, &x, &b) < 1e-10,
        "pipelined refactor solve residual"
    );
}

#[test]
fn klu_refactor_rejects_changed_pattern() {
    let a = circuit_like(60, 5);
    let mut s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    // Move one off-diagonal entry to a fresh position (same nnz).
    let (mut rows, mut cols, vals): (Vec<usize>, Vec<usize>, Vec<f64>) = {
        let mut rr = Vec::new();
        let mut cc = Vec::new();
        let mut vv = Vec::new();
        for j in 0..a.n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                rr.push(a.row_idx[k]);
                cc.push(j);
                vv.push(a.values[k]);
            }
        }
        (rr, cc, vv)
    };
    let moved = rows.iter().zip(&cols).position(|(&r, &c)| r != c).unwrap();
    rows[moved] = (rows[moved] + 1) % a.n;
    cols[moved] = (cols[moved] + 1) % a.n;
    let a2 = GeneralCsc::from_triplets(a.n, &rows, &cols, &vals).unwrap();
    if a2.nnz() != a.nnz() {
        return; // duplicate collapse: not the case under test
    }
    assert!(s.refactor(&a2).is_err(), "changed pattern must be rejected");
}

/// Max-norm relative residual of the *transposed* system `A^T x = b`.
fn resid_t<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
    resid(&a.transpose(), x, b)
}

#[test]
fn klu_solve_transpose_matches_factored_transpose() {
    // Reducible circuit-shaped matrix: solve_transpose on A's factors must
    // agree with a fresh factorization of A^T, and satisfy A^T x = b.
    let a = circuit_like(200, 42);
    let b: Vec<f64> = (0..200).map(|i| ((i * 3) % 13) as f64 - 6.0).collect();
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert!(s.n_blocks() >= 2, "bridge structure must be reducible");
    let x = s.solve_transpose(&b).unwrap();
    assert!(
        resid_t(&a, &x, &b) < 1e-12,
        "residual {}",
        resid_t(&a, &x, &b)
    );
    let st = KluSolver::factor(&a.transpose(), &KluSettings::default()).unwrap();
    let xr = st.solve(&b).unwrap();
    let diff = x
        .iter()
        .zip(&xr)
        .map(|(&p, &q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(
        diff < 1e-9,
        "transpose solve vs factored transpose differ by {diff}"
    );
}

#[test]
fn klu_solve_transpose_complex_plain_not_conjugate() {
    // Complex: solve_transpose must solve the PLAIN transpose A^T x = b
    // (adjoint convention: the caller conjugates for A^H). Off-diagonal
    // pivoting pressure included (small diagonal), as in the solve test.
    let c = |re, im| Complex::new(re, im);
    let m = 6;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(0.3, 0.05));
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(c(2.0, 0.3));
                rr.push(q);
                cc.push(p);
                vv.push(c(1.5, -0.2));
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(c(1.8, 0.1));
                rr.push(q);
                cc.push(p);
                vv.push(c(2.2, 0.4));
            }
        }
    }
    let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<Complex<f64>> = (0..n)
        .map(|i| c((i % 5) as f64 - 2.0, (i % 3) as f64))
        .collect();
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let x = s.solve_transpose(&b).unwrap();
    assert!(
        resid_t(&a, &x, &b) < 1e-12,
        "residual {}",
        resid_t(&a, &x, &b)
    );
    // A^H x = b via the documented conjugation recipe.
    let bc: Vec<Complex<f64>> = b.iter().map(|v| v.conj()).collect();
    let xh: Vec<Complex<f64>> = s
        .solve_transpose(&bc)
        .unwrap()
        .iter()
        .map(|v| v.conj())
        .collect();
    let ah = {
        let t = a.transpose();
        GeneralCsc::<Complex<f64>> {
            n: t.n,
            col_ptr: t.col_ptr.clone(),
            row_idx: t.row_idx.clone(),
            values: t.values.iter().map(|v| v.conj()).collect(),
        }
    };
    assert!(resid(&ah, &xh, &b) < 1e-12);
}

#[test]
fn klu_solve_transpose_singleton_blocks_and_options() {
    // Lower bidiagonal (all-singleton BTF blocks, pure F off-block path),
    // plus the no-BTF and no-scaling configurations on the circuit matrix.
    let n = 50;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(2.0 + (i % 3) as f64);
        if i + 1 < n {
            r.push(i + 1);
            c.push(i);
            v.push(-1.0);
        }
    }
    let tri = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
    let s = KluSolver::factor(&tri, &KluSettings::default()).unwrap();
    assert_eq!(s.n_blocks(), n);
    let b: Vec<f64> = (0..n).map(|i| i as f64 - 7.0).collect();
    let x = s.solve_transpose(&b).unwrap();
    assert!(resid_t(&tri, &x, &b) < 1e-14);

    let a = circuit_like(100, 21);
    let b: Vec<f64> = (0..a.n).map(|i| (i % 5) as f64 - 2.0).collect();
    for settings in [
        KluSettings::default().with_btf(false),
        KluSettings::default().with_row_scaling(false),
        KluSettings::default()
            .with_btf(false)
            .with_row_scaling(false),
    ] {
        let s = KluSolver::factor(&a, &settings).unwrap();
        let x = s.solve_transpose(&b).unwrap();
        assert!(resid_t(&a, &x, &b) < 1e-12, "settings {settings:?}");
    }
}

#[test]
fn klu_solve_transpose_after_refactor() {
    // The transpose solve must read the refactored values, not stale ones.
    let a = circuit_like(150, 99);
    let mut s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let a2 = {
        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        for j in 0..a.n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                rows.push(a.row_idx[k]);
                cols.push(j);
            }
        }
        let vals: Vec<f64> = a
            .values
            .iter()
            .enumerate()
            .map(|(k, &v)| v * (1.0 + 0.01 * ((k % 17) as f64)))
            .collect();
        GeneralCsc::from_triplets(a.n, &rows, &cols, &vals).unwrap()
    };
    s.refactor(&a2).unwrap();
    let b: Vec<f64> = (0..a.n).map(|i| (i % 9) as f64 - 4.0).collect();
    let x = s.solve_transpose(&b).unwrap();
    assert!(
        resid_t(&a2, &x, &b) < 1e-11,
        "residual {}",
        resid_t(&a2, &x, &b)
    );
}

#[test]
fn klu_solve_transpose_empty_and_dimension_check() {
    let a = GeneralCsc::<f64>::from_triplets(0, &[], &[], &[]).unwrap();
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert_eq!(s.solve_transpose(&[]).unwrap(), Vec::<f64>::new());
    let a = circuit_like(20, 1);
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert!(s.solve_transpose(&[0.0; 19]).is_err());
}

#[test]
fn klu_solve_many_matches_single() {
    let a = circuit_like(80, 3);
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let nrhs = 4;
    let b: Vec<f64> = (0..a.n * nrhs).map(|k| (k % 7) as f64 - 3.0).collect();
    let x = s.solve_many(&b, nrhs).unwrap();
    for col in 0..nrhs {
        let bc: Vec<f64> = (0..a.n).map(|i| b[col * a.n + i]).collect();
        let xc = s.solve(&bc).unwrap();
        for i in 0..a.n {
            assert_eq!(x[col * a.n + i], xc[i], "rhs {col} row {i}");
        }
    }
}

#[test]
fn klu_solve_refined_tightens_residual() {
    let a = circuit_like(120, 11);
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let b: Vec<f64> = (0..a.n).map(|i| ((i * i) % 23) as f64 - 11.0).collect();
    let x = s
        .solve_refined(&a, &b, &crate::RefinePolicy::steps(2))
        .unwrap()
        .0;
    assert!(resid(&a, &x, &b) < 1e-13);
}

#[test]
fn klu_without_btf_still_solves() {
    let a = circuit_like(100, 21);
    let s = KluSolver::factor(
        &a,
        &KluSettings {
            btf: false,
            ..KluSettings::default()
        },
    )
    .unwrap();
    assert_eq!(s.n_blocks(), 1);
    let b: Vec<f64> = (0..a.n).map(|i| (i % 5) as f64).collect();
    let x = s.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-12);
}

#[test]
fn klu_empty_matrix() {
    let a = GeneralCsc::<f64>::from_triplets(0, &[], &[], &[]).unwrap();
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert_eq!(s.solve(&[]).unwrap(), Vec::<f64>::new());
}

#[test]
fn klu_estimate_matches_actual_fill_on_dominant_matrix() {
    // Diagonally dominant -> threshold pivoting keeps every diagonal, so
    // the diagonal-pivot symbolic fill must be EXACT, and the estimate's
    // factor_nnz must equal the factored fill.
    let a = circuit_like(150, 33);
    let sym = KluSymbolic::analyze(&a, &KluSettings::default()).unwrap();
    let est = sym.estimate_memory::<f64>();
    let s = sym.factor(&a, &KluSettings::default()).unwrap();
    assert_eq!(est.factor_nnz as usize, s.factor_nnz());
    assert_eq!(sym.symbolic_factor_nnz(), s.factor_nnz());
    assert!(est.factor_flops > 0);
    assert_eq!(est.critical_path_flops, est.factor_flops);
    assert!(est.transient_peak_bytes >= est.factor_bytes);
}

#[test]
fn klu_diagnostics_phased_vs_oneshot() {
    let a = circuit_like(100, 4);
    let sym = KluSymbolic::analyze(&a, &KluSettings::default()).unwrap();
    // factor() never estimates implicitly: the estimate is attached only
    // when it was computed explicitly beforehand.
    let s0 = sym.factor(&a, &KluSettings::default()).unwrap();
    assert!(s0.diagnostics().estimate.is_none());
    let _ = sym.estimate_memory::<f64>();
    let s = sym.factor(&a, &KluSettings::default()).unwrap();
    let d = s.diagnostics();
    assert_eq!(d.threads, 1);
    assert_eq!(d.factor_nnz as usize, s.factor_nnz());
    assert!(d.estimate.is_some());
    assert_eq!(d.stages.len(), 1);
    assert_eq!(d.stages[0].name, "klu-factor");

    let mut s2 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    // the one-shot factor fills its diagnostics like the phased path
    assert_eq!(s2.diagnostics().stages.last().unwrap().name, "klu-factor");
    s2.refactor(&a).unwrap();
    assert_eq!(s2.diagnostics().stages.last().unwrap().name, "klu-refactor");
}

/// Cascaded stages with one-way inter-stage feeds: `stages` irreducible
/// diagonal blocks in the BTF (the reducible shape the Auto gate keys on).
fn cascaded(n: usize, stages: usize, seed: u64) -> GeneralCsc<f64> {
    let mut rng = Rng(seed | 1);
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let stage = n / stages;
    for j in 0..n {
        let s = (j / stage).min(stages - 1);
        let lo = s * stage;
        let hi = if s == stages - 1 { n } else { lo + stage };
        r.push(j);
        c.push(j);
        v.push(6.0 + rng.next_f64());
        // ring coupling inside the stage keeps the block irreducible
        let fwd = lo + (j - lo + 1) % (hi - lo);
        if fwd != j {
            r.push(fwd);
            c.push(j);
            v.push(-1.0 + 0.1 * rng.next_f64());
            r.push(j);
            c.push(fwd);
            v.push(-1.0 + 0.1 * rng.next_f64());
        }
        // one-way feed from the previous stage
        if s > 0 {
            r.push(j - stage);
            c.push(j);
            v.push(0.25 * rng.next_f64());
        }
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

#[test]
fn klu_parallel_auto_gate_resolves_from_structure() {
    // Small: below the nnz floor, Auto stays sequential.
    let a = cascaded(400, 6, 3);
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert!(!s.factors.parallel, "small case must stay sequential");
    // Multi-block and over the floor: Auto goes parallel.
    let a = cascaded(4000, 6, 9);
    let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    assert!(s.factors.block_ptr.len() > 4);
    assert!(a.nnz() >= 8_000);
    assert!(
        s.factors.parallel,
        "large multi-block case must parallelize"
    );
    // Off always wins.
    let s = KluSolver::factor(&a, &KluSettings::default().with_parallel(KluParallel::Off)).unwrap();
    assert!(!s.factors.parallel);
}

#[test]
fn klu_parallel_factor_bit_identical() {
    let a = circuit_like(600, 7);
    let sym = KluSymbolic::analyze(&a, &KluSettings::default()).unwrap();
    let s1 = sym
        .factor(&a, &KluSettings::default().with_parallel(KluParallel::Off))
        .unwrap();
    let s2 = sym
        .factor(&a, &KluSettings::default().with_parallel(KluParallel::On))
        .unwrap();
    assert!(s2.factors.block_ptr.len() > 2, "needs a multi-block case");
    assert_eq!(s1.factors.l_rowidx, s2.factors.l_rowidx);
    assert_eq!(s1.factors.u_rowidx, s2.factors.u_rowidx);
    assert_eq!(s1.factors.row_perm, s2.factors.row_perm);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&s1.factors.l_val), bits(&s2.factors.l_val));
    assert_eq!(bits(&s1.factors.u_val), bits(&s2.factors.u_val));
    assert_eq!(bits(&s1.factors.udiag), bits(&s2.factors.udiag));
    assert_eq!(s1.factors.scatter_expect, s2.factors.scatter_expect);
    assert_eq!(s1.factors.scatter_target, s2.factors.scatter_target);
    let b: Vec<f64> = (0..a.n).map(|i| (i % 5) as f64 - 2.0).collect();
    let x1 = s1.solve(&b).unwrap();
    let x2 = s2.solve(&b).unwrap();
    assert_eq!(bits(&x1), bits(&x2));

    // Refactor honors the same opt-in and stays bit-identical too.
    let a2 = GeneralCsc {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: a.values.iter().map(|&v| v * 1.25).collect(),
    };
    let mut s1 = s1;
    let mut s2 = s2;
    s1.refactor(&a2).unwrap();
    s2.refactor(&a2).unwrap();
    assert_eq!(bits(&s1.factors.l_val), bits(&s2.factors.l_val));
    assert_eq!(bits(&s1.factors.u_val), bits(&s2.factors.u_val));
    assert_eq!(bits(&s1.factors.udiag), bits(&s2.factors.udiag));
    let y1 = s1.solve(&b).unwrap();
    let y2 = s2.solve(&b).unwrap();
    assert_eq!(bits(&y1), bits(&y2));
}

#[test]
fn klu_composes_as_gmres_preconditioner() {
    use crate::numeric::krylov::gmres;
    let a = circuit_like(120, 55);
    let m = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let b: Vec<f64> = (0..a.n).map(|i| (i % 7) as f64 - 3.0).collect();
    // Exact preconditioner -> GMRES converges in one iteration.
    let res = gmres(
        &a,
        &b,
        &m,
        &crate::KrylovSettings::default()
            .with_tol(1e-12)
            .with_max_iter(5)
            .with_restart(5),
        None,
    )
    .unwrap();
    assert!(res.converged);
    assert!(res.iters <= 2, "iterations {}", res.iters);
    assert!(resid(&a, &res.x, &b) < 1e-10);
}

#[test]
fn klu_settings_compose() {
    let s = KluSettings::default()
        .with_pivot_threshold(1.0)
        .with_row_scaling(false)
        .with_btf(false);
    assert_eq!(s.pivot_threshold, 1.0);
    assert!(!s.row_scaling);
    assert!(!s.btf);
    let a = circuit_like(80, 9);
    let solver = KluSolver::factor(&a, &s).unwrap();
    let b: Vec<f64> = (0..a.n).map(|i| (i % 3) as f64).collect();
    let x = solver.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-12);
}

/// The exported factors reproduce the factored matrix, `P_r R A P_c =
/// L U + F`, with `L` and `U` inside the diagonal blocks and `F` above.
#[test]
fn exported_factors_reproduce_the_matrix() {
    for (seed, scaling, btf) in [(10, true, true), (20, false, true), (30, true, false)] {
        let a = cascaded(60, 4, seed);
        let s = KluSettings::default()
            .with_row_scaling(scaling)
            .with_btf(btf);
        let solver = KluSolver::factor(&a, &s).unwrap();
        let (l, u, f) = (solver.l_matrix(), solver.u_matrix(), solver.f_matrix());
        let bp = solver.block_ptr();
        let block = |i: usize| bp.partition_point(|&b| b <= i) - 1;
        for (m, name) in [(&l, "L"), (&u, "U"), (&f, "F")] {
            m.validate().unwrap();
            for j in 0..m.n {
                for &i in &m.row_idx[m.col_ptr[j]..m.col_ptr[j + 1]] {
                    let inside = match name {
                        "L" => i >= j && block(i) == block(j),
                        "U" => i <= j && block(i) == block(j),
                        _ => block(i) < block(j),
                    };
                    assert!(inside, "{name} entry ({i}, {j}) out of place");
                }
            }
        }
        assert!((0..l.n).all(|j| l.row_idx[l.col_ptr[j]] == j && l.values[l.col_ptr[j]] == 1.0));
        // (L U + F) x against P_r R A P_c x.
        let n = a.n;
        let mut rng = Rng(seed | 99);
        let x: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let (mut ux, mut lux, mut fx) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        u.matvec(&x, &mut ux);
        l.matvec(&ux, &mut lux);
        f.matvec(&x, &mut fx);
        let mut xc = vec![0.0; n];
        for (k, &c) in solver.col_perm().iter().enumerate() {
            xc[c] = x[k];
        }
        let mut ax = vec![0.0; n];
        a.matvec(&xc, &mut ax);
        let rs = solver.row_scale();
        for (k, &r) in solver.row_perm().iter().enumerate() {
            assert!((lux[k] + fx[k] - rs[r] * ax[r]).abs() < 1e-12, "row {k}");
        }
    }
}

/// The exported factors of a complex matrix reproduce it too.
#[test]
fn exported_complex_factors_reproduce_the_matrix() {
    use num_complex::Complex;
    type C = Complex<f64>;
    let ar = cascaded(60, 4, 40);
    let a = GeneralCsc::<C> {
        n: ar.n,
        col_ptr: ar.col_ptr.clone(),
        row_idx: ar.row_idx.clone(),
        values: ar
            .values
            .iter()
            .enumerate()
            .map(|(k, &v)| C::new(v, 0.3 * v * ((k % 3) as f64 - 1.0)))
            .collect(),
    };
    let solver = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let (l, u, f) = (solver.l_matrix(), solver.u_matrix(), solver.f_matrix());
    let n = a.n;
    let x: Vec<C> = (0..n)
        .map(|i| C::new(i as f64 * 0.1, 1.0 - i as f64 * 0.02))
        .collect();
    let zero = C::new(0.0, 0.0);
    let (mut ux, mut lux, mut fx) = (vec![zero; n], vec![zero; n], vec![zero; n]);
    u.matvec(&x, &mut ux);
    l.matvec(&ux, &mut lux);
    f.matvec(&x, &mut fx);
    let mut xc = vec![zero; n];
    for (k, &c) in solver.col_perm().iter().enumerate() {
        xc[c] = x[k];
    }
    let mut ax = vec![zero; n];
    a.matvec(&xc, &mut ax);
    let rs = solver.row_scale();
    for (k, &r) in solver.row_perm().iter().enumerate() {
        assert!((lux[k] + fx[k] - ax[r] * rs[r]).norm() < 1e-12, "row {k}");
    }
}
