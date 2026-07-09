//! Smoke tests for the public AMF surface.
//!
//! Mirrors the five fixtures in `rslab-ordering-core/tests/invariants.rs`
//! but exercises them through the public `rslab-amf` API rather than
//! the bare `order::<MinFill>` driver. The contract is the same:
//! every fixture produces a permutation of `0..n`, and the
//! diagnostic counters surface without panicking.

use rslab_amf::{amf_order, amf_order_full, amf_order_with_stats, AmfOptions, CscPattern};

fn arrow_3() -> (Vec<i32>, Vec<i32>) {
    let col_ptr = vec![0, 3, 5, 7];
    let row_idx = vec![0, 1, 2, 0, 1, 0, 2];
    (col_ptr, row_idx)
}

fn dual_arrow_5() -> (Vec<i32>, Vec<i32>) {
    let col_ptr = vec![0, 5, 8, 11, 14, 19];
    let row_idx = vec![
        0, 1, 2, 3, 4, // col 0
        0, 1, 4, // col 1
        0, 2, 4, // col 2
        0, 3, 4, // col 3
        0, 1, 2, 3, 4, // col 4
    ];
    (col_ptr, row_idx)
}

fn tridiag(n: usize) -> (Vec<i32>, Vec<i32>) {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        if j > 0 {
            row_idx.push((j - 1) as i32);
        }
        row_idx.push(j as i32);
        if j + 1 < n {
            row_idx.push((j + 1) as i32);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx)
}

fn banded(n: usize, b: usize) -> (Vec<i32>, Vec<i32>) {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        let lo = j.saturating_sub(b);
        let hi = (j + b + 1).min(n);
        for r in lo..hi {
            row_idx.push(r as i32);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx)
}

fn assert_is_permutation(perm: &[i32], n: usize) {
    assert_eq!(perm.len(), n);
    let mut seen = vec![false; n];
    for &p in perm {
        assert!(p >= 0 && (p as usize) < n);
        assert!(!seen[p as usize]);
        seen[p as usize] = true;
    }
    assert!(seen.iter().all(|&b| b));
}

#[test]
fn amf_order_arrow_3() {
    let (cp, ri) = arrow_3();
    let p = CscPattern::new(3, &cp, &ri).expect("valid pattern");
    let perm = amf_order(&p).expect("amf_order succeeds");
    assert_is_permutation(&perm, 3);
}

#[test]
fn amf_order_dual_arrow_5() {
    let (cp, ri) = dual_arrow_5();
    let p = CscPattern::new(5, &cp, &ri).expect("valid pattern");
    let perm = amf_order(&p).expect("amf_order succeeds");
    assert_is_permutation(&perm, 5);
}

#[test]
fn amf_order_tridiag_10() {
    let (cp, ri) = tridiag(10);
    let p = CscPattern::new(10, &cp, &ri).expect("valid pattern");
    let perm = amf_order(&p).expect("amf_order succeeds");
    assert_is_permutation(&perm, 10);
}

#[test]
fn amf_order_banded_20_3() {
    let (cp, ri) = banded(20, 3);
    let p = CscPattern::new(20, &cp, &ri).expect("valid pattern");
    let perm = amf_order(&p).expect("amf_order succeeds");
    assert_is_permutation(&perm, 20);
}

#[test]
fn amf_order_empty() {
    let cp = [0i32];
    let ri: [i32; 0] = [];
    let p = CscPattern::new(0, &cp, &ri).expect("empty pattern");
    let perm = amf_order(&p).expect("n=0 succeeds");
    assert!(perm.is_empty());
}

#[test]
fn amf_order_full_returns_three_tuple() {
    let (cp, ri) = tridiag(5);
    let p = CscPattern::new(5, &cp, &ri).expect("valid pattern");
    let (perm, ord_stats, _amf_stats) =
        amf_order_full(&p, &AmfOptions::default()).expect("amf_order_full succeeds");
    assert_is_permutation(&perm, 5);
    // time_us is u64; reading it must not panic. Allow zero on
    // very fast machines / micro-fixtures.
    let _ = ord_stats.time_us;
    assert!(ord_stats.fill_estimate.is_none());
    assert!(ord_stats.flop_estimate.is_none());
}

#[test]
fn amf_order_with_stats_returns_amf_stats() {
    let (cp, ri) = banded(10, 2);
    let p = CscPattern::new(10, &cp, &ri).expect("valid pattern");
    let (perm, stats) = amf_order_with_stats(&p).expect("amf_order_with_stats succeeds");
    assert_is_permutation(&perm, 10);
    // Banded matrix has no dense rows; deferral count is zero.
    assert_eq!(stats.n_dense_deferred, 0);
}
