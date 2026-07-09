//! Reverse Cuthill-McKee (RCM) band/profile-reducing ordering.
//!
//! RCM renumbers a symmetric graph so the nonzeros cluster tightly around the
//! diagonal (small bandwidth and profile). On banded / structured matrices
//! (stencils, structured FEM) where nested dissection over-separates and
//! minimum-degree scatters fill, the band factor of an RCM ordering has less
//! fill and factors faster. It is the classic George-Liu construction: a
//! degree-sorted breadth-first sweep from a pseudo-peripheral start vertex,
//! reversed (Cuthill & McKee 1969; George & Liu 1981, *Computer Solution of
//! Large Sparse Positive Definite Systems*).
//!
//! Convention (the shared ordering contract): the returned `perm` is
//! new-to-old, `perm[k] = j` meaning new index `k` is original vertex `j`.

use crate::{CscPattern, OrderingError};
use std::collections::VecDeque;

/// Compute a Reverse Cuthill-McKee ordering of the symmetric pattern.
///
/// Returns the new-to-old permutation `perm` (`perm[k]` is the original vertex
/// placed at new position `k`), reducing the bandwidth and profile of `A`.
/// Disconnected components are ordered one after another. The diagonal is
/// ignored (self-loops are not edges).
///
/// # Errors
///
/// Returns [`OrderingError::IndexOverflow`] if `n` does not fit in `i32`
/// (the contract's index type). Never fails on a well-formed [`CscPattern`].
pub fn rcm_order(pattern: &CscPattern<'_>) -> Result<Vec<i32>, OrderingError> {
    let n = pattern.n;
    if i32::try_from(n).is_err() {
        return Err(OrderingError::IndexOverflow);
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let col_ptr = pattern.col_ptr;
    let row_idx = pattern.row_idx;

    // Off-diagonal degree of every vertex (self-loops excluded).
    let mut deg = vec![0i32; n];
    for j in 0..n {
        let (s, e) = (col_ptr[j] as usize, col_ptr[j + 1] as usize);
        let mut d = 0i32;
        for &r in &row_idx[s..e] {
            if r as usize != j {
                d += 1;
            }
        }
        deg[j] = d;
    }

    // BFS distance buffer, reset (only for touched vertices) after each sweep.
    let mut dist = vec![-1i32; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut visited = vec![false; n];
    // Cuthill-McKee order (all components concatenated), reversed at the end.
    let mut cm: Vec<i32> = Vec::with_capacity(n);
    let mut queue: VecDeque<usize> = VecDeque::new();

    for seed in 0..n {
        if visited[seed] {
            continue;
        }
        let start = pseudo_peripheral(seed, col_ptr, row_idx, &deg, &mut dist, &mut touched);

        // Degree-sorted BFS from `start` (the Cuthill-McKee sweep): each dequeued
        // vertex appends its not-yet-visited neighbours in ascending-degree order.
        visited[start] = true;
        queue.clear();
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            cm.push(v as i32);
            let (s, e) = (col_ptr[v] as usize, col_ptr[v + 1] as usize);
            let mut nbrs: Vec<usize> = row_idx[s..e]
                .iter()
                .map(|&r| r as usize)
                .filter(|&w| w != v && !visited[w])
                .collect();
            nbrs.sort_unstable_by_key(|&w| deg[w]);
            for w in nbrs {
                visited[w] = true;
                queue.push_back(w);
            }
        }
    }

    debug_assert_eq!(cm.len(), n, "every vertex ordered exactly once");
    // Reverse the Cuthill-McKee order to get RCM (smaller profile).
    cm.reverse();
    Ok(cm)
}

/// George-Liu pseudo-peripheral start vertex for the component containing
/// `seed`: iterate rooted level structures, hopping to a minimum-degree vertex
/// of the last level while the eccentricity keeps growing. Uses `dist` as
/// scratch and restores it (via `touched`) before returning.
fn pseudo_peripheral(
    seed: usize,
    col_ptr: &[i32],
    row_idx: &[i32],
    deg: &[i32],
    dist: &mut [i32],
    touched: &mut Vec<usize>,
) -> usize {
    let mut u = seed;
    let (mut ecc, mut last) = level_structure(u, col_ptr, row_idx, dist, touched);
    // Bounded: George-Liu terminates in a handful of hops; cap for safety.
    for _ in 0..16 {
        // Candidate: minimum-degree vertex in the last (deepest) level.
        let cand = *last
            .iter()
            .min_by_key(|&&w| deg[w])
            .expect("last level is non-empty");
        let (ecc2, last2) = level_structure(cand, col_ptr, row_idx, dist, touched);
        if ecc2 > ecc {
            u = cand;
            ecc = ecc2;
            last = last2;
        } else {
            break;
        }
    }
    u
}

/// Rooted level structure of a BFS from `start`: returns the eccentricity
/// (deepest level index) and the set of vertices in that deepest level. Fills
/// and then restores `dist` (touched vertices reset to `-1`).
fn level_structure(
    start: usize,
    col_ptr: &[i32],
    row_idx: &[i32],
    dist: &mut [i32],
    touched: &mut Vec<usize>,
) -> (i32, Vec<usize>) {
    touched.clear();
    let mut q: VecDeque<usize> = VecDeque::new();
    dist[start] = 0;
    touched.push(start);
    q.push_back(start);
    let mut ecc = 0i32;
    while let Some(v) = q.pop_front() {
        let dv = dist[v];
        if dv > ecc {
            ecc = dv;
        }
        let (s, e) = (col_ptr[v] as usize, col_ptr[v + 1] as usize);
        for &r in &row_idx[s..e] {
            let w = r as usize;
            if w != v && dist[w] < 0 {
                dist[w] = dv + 1;
                touched.push(w);
                q.push_back(w);
            }
        }
    }
    let last: Vec<usize> = touched
        .iter()
        .copied()
        .filter(|&w| dist[w] == ecc)
        .collect();
    for &w in touched.iter() {
        dist[w] = -1;
    }
    (ecc, last)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a full-symmetric CSC pattern from an edge list (both directions
    /// added; diagonal included) for the ordering contract.
    fn pattern_from_edges(n: usize, edges: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for v in 0..n {
            adj[v].push(v);
        }
        for &(a, b) in edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        let mut col_ptr = vec![0i32];
        let mut row_idx = Vec::new();
        for v in 0..n {
            adj[v].sort_unstable();
            adj[v].dedup();
            for &w in &adj[v] {
                row_idx.push(w as i32);
            }
            col_ptr.push(row_idx.len() as i32);
        }
        (col_ptr, row_idx)
    }

    /// Bandwidth of `A` under a given new-to-old ordering: max |new(i) - new(j)|
    /// over edges (i, j).
    fn bandwidth(n: usize, edges: &[(usize, usize)], perm: &[i32]) -> usize {
        let mut inv = vec![0usize; n];
        for (newpos, &old) in perm.iter().enumerate() {
            inv[old as usize] = newpos;
        }
        let mut bw = 0usize;
        for &(a, b) in edges {
            let d = inv[a].abs_diff(inv[b]);
            if d > bw {
                bw = d;
            }
        }
        bw
    }

    #[test]
    fn rcm_reduces_bandwidth_of_reversed_path() {
        // A path graph numbered to maximise bandwidth: connect i to n-1-i's
        // neighbourhood via a "reflected" chain. RCM must recover a near-1
        // bandwidth (a path is a band-1 graph under its natural order).
        let n = 200;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        // Scramble: relabel vertex i -> (i * 97) mod n (a bijection) so the
        // stored order is far from banded.
        let relabel: Vec<usize> = (0..n).map(|i| (i * 97) % n).collect();
        let scrambled: Vec<(usize, usize)> = edges
            .iter()
            .map(|&(a, b)| (relabel[a], relabel[b]))
            .collect();
        let (col_ptr, row_idx) = pattern_from_edges(n, &scrambled);
        let pat = CscPattern::new(n, &col_ptr, &row_idx).unwrap();

        let identity: Vec<i32> = (0..n as i32).collect();
        let bw_before = bandwidth(n, &scrambled, &identity);
        let perm = rcm_order(&pat).unwrap();
        let bw_after = bandwidth(n, &scrambled, &perm);

        assert_eq!(perm.len(), n);
        // A valid permutation.
        let mut seen = vec![false; n];
        for &p in &perm {
            assert!(!seen[p as usize], "duplicate index");
            seen[p as usize] = true;
        }
        // RCM recovers a small bandwidth from a scrambled path.
        assert!(
            bw_after <= 2,
            "RCM bandwidth {bw_after} (was {bw_before}) should be near 1 on a path"
        );
    }

    #[test]
    fn rcm_2d_grid_beats_natural_scramble() {
        // 5-point stencil on an m x m grid, vertices relabelled by a bijection.
        let m = 20;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let mut edges = Vec::new();
        for a in 0..m {
            for b in 0..m {
                if b + 1 < m {
                    edges.push((idx(a, b), idx(a, b + 1)));
                }
                if a + 1 < m {
                    edges.push((idx(a, b), idx(a + 1, b)));
                }
            }
        }
        let relabel: Vec<usize> = (0..n).map(|i| (i * 131) % n).collect();
        let scrambled: Vec<(usize, usize)> = edges
            .iter()
            .map(|&(a, b)| (relabel[a], relabel[b]))
            .collect();
        let (col_ptr, row_idx) = pattern_from_edges(n, &scrambled);
        let pat = CscPattern::new(n, &col_ptr, &row_idx).unwrap();

        let identity: Vec<i32> = (0..n as i32).collect();
        let bw_before = bandwidth(n, &scrambled, &identity);
        let perm = rcm_order(&pat).unwrap();
        let bw_after = bandwidth(n, &scrambled, &perm);
        // The grid's optimal bandwidth is ~m; RCM must get within a small factor
        // and hugely beat the scrambled numbering.
        assert!(
            bw_after < bw_before / 2 && bw_after <= 3 * m,
            "RCM grid bandwidth {bw_after} (was {bw_before}, m={m})"
        );
    }

    #[test]
    fn rcm_handles_disconnected_and_empty() {
        assert!(rcm_order(&CscPattern::new(0, &[0], &[]).unwrap())
            .unwrap()
            .is_empty());
        // Two disjoint edges: 0-1 and 2-3.
        let (col_ptr, row_idx) = pattern_from_edges(4, &[(0, 1), (2, 3)]);
        let pat = CscPattern::new(4, &col_ptr, &row_idx).unwrap();
        let perm = rcm_order(&pat).unwrap();
        assert_eq!(perm.len(), 4);
        let mut seen = vec![false; 4];
        for &p in &perm {
            seen[p as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "all vertices ordered");
    }
}
