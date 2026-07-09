//! Regression test for issue #3: ScotchND used to silently fall back
//! to AMD on KKT-shaped matrices, then was fixed by O13.
//!
//! Builds a small PoissonControl KKT *pattern* (n_kkt = 3K²), runs
//! `rslab_scotch::scotch_order_full` and `rslab_amd::amd_order` on the
//! same full-symmetric pattern, and compares them.
//!
//! History: originally this test *locked in* the degenerate symptom -
//! the vertex-separator FM stopped as soon as both priority-queue heads
//! were imbalance-rejected (the O13 bug), leaving a one-sided bisection;
//! `node_nd` then hit its `a_verts.is_empty() || b_verts.is_empty()`
//! guard and emitted a whole-graph `amd_leaf`, so the ScotchND
//! permutation was byte-equal to AMD's (`n_separator_vertices == 0`)
//! while the caller still reported `ScotchND`.
//!
//! O13 lets FM skip the infeasible heads and keep refining, so the
//! bisection is now balanced and genuine nested dissection proceeds.
//! This test now guards against regressing back to that degenerate
//! path. (It does not by itself close issue #3 - the upper-layer
//! visibility/fallback machinery is unaffected - but the specific KKT
//! degeneracy that motivated the issue no longer reproduces here.)

use rslab_amd::amd_order;
use rslab_ordering_core::CscPattern;
use rslab_scotch::{scotch_order_full, ScotchOptions};

use std::collections::BTreeSet;

fn poisson_kkt_pattern(k: usize) -> (Vec<i32>, Vec<i32>, usize) {
    // Build the full-symmetric pattern (with diagonal) of the Poisson
    // optimal-control KKT system from `src/bin/diag_poisson_kkt.rs`.
    // Only the pattern matters for the ordering call.
    let m = k * k;
    let n = 3 * m;
    let mut s: BTreeSet<(i32, i32)> = BTreeSet::new();

    for i in 0..n {
        s.insert((i as i32, i as i32));
    }

    for i in 0..k {
        for j in 0..k {
            let c = i * k + j;
            let con_row = (2 * m + c) as i32;
            // 5-point stencil couples constraint row to u block
            let center = c as i32;
            s.insert((con_row, center));
            s.insert((center, con_row));
            if i > 0 {
                let nbr = ((i - 1) * k + j) as i32;
                s.insert((con_row, nbr));
                s.insert((nbr, con_row));
            }
            if i + 1 < k {
                let nbr = ((i + 1) * k + j) as i32;
                s.insert((con_row, nbr));
                s.insert((nbr, con_row));
            }
            if j > 0 {
                let nbr = (i * k + (j - 1)) as i32;
                s.insert((con_row, nbr));
                s.insert((nbr, con_row));
            }
            if j + 1 < k {
                let nbr = (i * k + (j + 1)) as i32;
                s.insert((con_row, nbr));
                s.insert((nbr, con_row));
            }
            // f coupling: con_row <-> (m + c)
            let f = (m + c) as i32;
            s.insert((con_row, f));
            s.insert((f, con_row));
        }
    }

    let mut col_ptr: Vec<i32> = vec![0];
    let mut row_idx: Vec<i32> = Vec::new();
    let mut by_col: Vec<Vec<i32>> = vec![Vec::new(); n];
    for (r, c) in s {
        by_col[c as usize].push(r);
    }
    for col in &mut by_col {
        col.sort();
    }
    for col in &by_col {
        for &r in col {
            row_idx.push(r);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx, n)
}

#[test]
fn issue_3_scotch_recurses_on_kkt_after_o13() {
    // Guards the post-O13 behavior on a KKT pattern: the
    // vertex-separator FM no longer stops early at imbalance-rejected
    // heads, so the top-level bisection is balanced and genuine nested
    // dissection runs instead of degenerating into a whole-graph AMD
    // leaf. See the module doc for the pre-O13 history.
    let k = 20; // n_kkt = 1200; nnz_per_row ~ 4.9 on full-symmetric
    let (col_ptr, row_idx, n) = poisson_kkt_pattern(k);
    let pat = CscPattern::new(n, &col_ptr, &row_idx).expect("pattern valid");

    let amd_perm = amd_order(&pat).expect("amd ok");
    let (scotch_perm, _ostats, sstats) =
        scotch_order_full(&pat, &ScotchOptions::default()).expect("scotch ok");

    eprintln!("issue #3 (post-O13): n={}, scotch stats={:?}", n, sstats);

    // Post-O13: bisection is no longer degenerate, so ND produces a
    // genuine separator. (Pre-O13 this was 0 - a whole-graph fallback.)
    assert!(
        sstats.n_separator_vertices > 0,
        "post-O13 ScotchND must produce a real separator on KKT (no \
         degenerate one-sided bisection); got {}",
        sstats.n_separator_vertices
    );
    // Because real ND ran, the permutation diverges from the pure-AMD
    // fallback the degenerate path used to emit byte-for-byte.
    assert_ne!(
        amd_perm, scotch_perm,
        "post-O13 ScotchND should no longer be byte-equal to AMD on KKT"
    );
}
