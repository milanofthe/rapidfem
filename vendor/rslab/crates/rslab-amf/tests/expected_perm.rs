//! Hand-derived expected-perm fixtures for AMF (Phase B.4).
//!
//! Each fixture encodes a *qualitative* property of the AMF metric
//! that follows directly from the score definition
//! `RMF(i) = (deg(i)*(deg(i)-1+2*degme) - WF(i)) / (nv(i)+1)`
//! at iteration 0, when no element has been formed yet and the
//! score reduces to the initial degree.
//!
//! The plan (`dev/plans/amf-clean-room.md` Phase B deliverable 6)
//! sketches three fixtures. For each, only the assertion that is
//! *defensibly derivable from the metric without simulating the
//! quotient-graph dynamics* is pinned here. Tighter pins (full perm
//! match) are deferred to Phase C, where the MUMPS HAMF4 oracle
//! will provide the external reference. Weaker, metric-only
//! claims:
//!
//! 1. 3x3 arrowhead -- the unique-max-degree hub vertex is
//!    eliminated last. (Iteration 0 picks min-deg; iterations 1+
//!    keep the hub at strict max-deg among survivors.)
//! 2. 5x5 dual-arrowhead -- at iteration 0 the metric strictly
//!    prefers a spine vertex (deg=2) over either hub (deg=4).
//! 3. 7x7 tridiagonal -- at iteration 0 the metric strictly prefers
//!    an endpoint (deg=1) over any interior vertex (deg=2).
//!
//! Reference: Amestoy (1999) habilitation thesis. The first-
//! iteration claim is immediate: with no elements formed,
//! `degme = 0` and `WF(i) = 0`, so `RMF(i) = deg(i)*(deg(i)-1) /
//! (nv(i)+1) = (deg(i)^2 - deg(i)) / 1` (all `nv(i) = 0` at init).
//! This is monotone-increasing in `deg(i)` over deg >= 1, so the
//! min-`deg` vertex is the strict argmin over score.

use rslab_amf::{amf_order, CscPattern};

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

fn position_of(perm: &[i32], v: i32) -> usize {
    perm.iter()
        .position(|&p| p == v)
        .unwrap_or_else(|| panic!("vertex {v} not in perm"))
}

#[test]
fn amf_arrow_3_hub_last() {
    // Vertex 0 is the unique-max-degree hub (deg=2 vs 1, 1 for the
    // leaves). At iteration 0 the leaves are picked first. After
    // either of {1, 2} is eliminated, the hub remains adjacent to
    // the surviving leaf via a freshly-formed element, but its
    // residual neighbour count is still strictly above the
    // surviving leaf's. Therefore the hub is eliminated last.
    let (cp, ri) = arrow_3();
    let p = CscPattern::new(3, &cp, &ri).expect("valid pattern");
    let perm = amf_order(&p).expect("amf_order succeeds");
    assert_eq!(perm.len(), 3);
    assert_eq!(
        position_of(&perm, 0),
        2,
        "AMF must place hub vertex 0 last; got perm = {perm:?}"
    );
}

#[test]
fn amf_dual_arrow_5_first_pick_is_spine() {
    // Vertices 0 and 4 are hubs (deg=4); 1, 2, 3 are spine (deg=2).
    // At iteration 0 the AMF score is `deg*(deg-1)` (degme=0,
    // WF=0, nv=0), so spine score = 2 < hub score = 12. Therefore
    // the first pivot must be a spine vertex (one of 1, 2, 3).
    //
    // The stronger claim "both hubs deferred to the last two
    // positions" depends on score arithmetic at iteration 1 onward
    // (where one hub may achieve a lower fill score than a
    // surviving spine, depending on quantization). That assertion
    // is deferred to Phase C with the MUMPS HAMF4 oracle.
    let (cp, ri) = dual_arrow_5();
    let p = CscPattern::new(5, &cp, &ri).expect("valid pattern");
    let perm = amf_order(&p).expect("amf_order succeeds");
    assert_eq!(perm.len(), 5);
    let first = perm[0];
    assert!(
        first == 1 || first == 2 || first == 3,
        "AMF first pivot on dual-arrowhead must be a spine vertex (1, 2, or 3); \
         got perm = {perm:?}"
    );
    // The strict-max hub at the final iteration is also pinnable
    // structurally: at least one of the two hubs occupies the last
    // position (the strict-max-deg survivor at iteration n-1 must
    // be unique because hubs are connected to each other through
    // every previously-formed element).
    assert!(
        perm[4] == 0 || perm[4] == 4,
        "AMF last pivot on dual-arrowhead must be a hub vertex (0 or 4); \
         got perm = {perm:?}"
    );
}

#[test]
fn amf_tridiag_7_first_pick_is_endpoint() {
    // Endpoints 0 and 6 have deg=1; interior vertices 1..=5 have
    // deg=2. At iteration 0 endpoint score = 0 (deg*(deg-1) = 0),
    // strictly less than interior score = 2. Therefore the first
    // pivot must be an endpoint.
    //
    // The stronger claim "both endpoints picked before any
    // interior" does not hold under standard quotient-graph
    // tie-breaking: after eliminating one endpoint, the adjacent
    // interior vertex acquires deg=1 in the residual graph and
    // can be picked before the other endpoint. This is the
    // expected sweep-from-one-end behaviour shared with AMD; it is
    // not a regression. Pin only the first-pivot claim.
    let (cp, ri) = tridiag(7);
    let p = CscPattern::new(7, &cp, &ri).expect("valid pattern");
    let perm = amf_order(&p).expect("amf_order succeeds");
    assert_eq!(perm.len(), 7);
    let first = perm[0];
    assert!(
        first == 0 || first == 6,
        "AMF first pivot on tridiag(7) must be an endpoint (0 or 6); \
         got perm = {perm:?}"
    );
}
