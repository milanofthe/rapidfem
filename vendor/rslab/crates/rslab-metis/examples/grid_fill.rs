//! Fill experiment harness on the 40^3 7-point grid (the PARDISO
//! reference pattern, helmholtz 40^3). Computes exact scalar
//! nnz(L) (elimination-tree column counts, no supernode padding) for
//! metis_order_full under varying options, plus an AMD baseline.
//!
//! Reference points (factor_probe_helmholtz, symbolic incl. padding):
//! rslab MetisND 25.06 M, AMF 24.7 M, MKL PARDISO ND 12.9 M.
//!
//! `cargo run --release -p rslab-metis --example grid_fill`

use rslab_metis::{metis_order_full, CscPattern, MetisOptions};

/// 7-point grid pattern, full symmetric, diagonal included.
fn grid3d_pattern(m: usize) -> (usize, Vec<i32>, Vec<i32>) {
    let idx = |x: usize, y: usize, z: usize| ((z * m + y) * m + x) as i32;
    let n = m * m * m;
    let mut cols: Vec<Vec<i32>> = vec![Vec::new(); n];
    for z in 0..m {
        for y in 0..m {
            for x in 0..m {
                let k = idx(x, y, z);
                let mut push = |a: i32, b: i32| {
                    cols[a as usize].push(b);
                    cols[b as usize].push(a);
                };
                if x + 1 < m {
                    push(k, idx(x + 1, y, z));
                }
                if y + 1 < m {
                    push(k, idx(x, y + 1, z));
                }
                if z + 1 < m {
                    push(k, idx(x, y, z + 1));
                }
            }
        }
    }
    let mut col_ptr: Vec<i32> = Vec::with_capacity(n + 1);
    let mut row_idx: Vec<i32> = Vec::new();
    col_ptr.push(0);
    for (c, col) in cols.iter_mut().enumerate() {
        col.push(c as i32);
        col.sort_unstable();
        row_idx.extend_from_slice(col);
        col_ptr.push(row_idx.len() as i32);
    }
    (n, col_ptr, row_idx)
}

/// Exact nnz(L) (incl. diagonal) and flop proxy sum(cc^2) for the
/// permuted pattern. `perm` is new-to-old.
fn fill_of_perm(n: usize, col_ptr: &[i32], row_idx: &[i32], perm: &[i32]) -> (u64, f64) {
    let mut iperm: Vec<i32> = vec![0; n];
    for (new, &old) in perm.iter().enumerate() {
        iperm[old as usize] = new as i32;
    }
    // Elimination tree via ancestor path compression (Davis alg. 4.1).
    let mut parent: Vec<i32> = vec![-1; n];
    let mut ancestor: Vec<i32> = vec![-1; n];
    let mut lower: Vec<Vec<i32>> = vec![Vec::new(); n]; // permuted strict lower adjacency per row i
    for i in 0..n {
        let old = perm[i] as usize;
        let lo = col_ptr[old] as usize;
        let hi = col_ptr[old + 1] as usize;
        for &u in &row_idx[lo..hi] {
            let j = iperm[u as usize];
            if (j as usize) < i {
                lower[i].push(j);
            }
        }
        for &j in &lower[i] {
            let mut jj = j as usize;
            loop {
                let a = ancestor[jj];
                if a == i as i32 {
                    break;
                }
                ancestor[jj] = i as i32;
                if a == -1 {
                    parent[jj] = i as i32;
                    break;
                }
                jj = a as usize;
            }
        }
    }
    // Column counts via row-subtree traversal.
    let mut cc: Vec<u32> = vec![1; n];
    let mut mark: Vec<i32> = vec![-1; n];
    for i in 0..n {
        mark[i] = i as i32;
        for &j in &lower[i] {
            let mut jj = j as usize;
            while mark[jj] != i as i32 {
                cc[jj] += 1;
                mark[jj] = i as i32;
                jj = parent[jj] as usize;
            }
        }
    }
    let nnz: u64 = cc.iter().map(|&c| c as u64).sum();
    let flops: f64 = cc.iter().map(|&c| (c as f64) * (c as f64)).sum();
    (nnz, flops)
}

/// Oracle: geometric nested dissection on the m^3 grid. Cuts the
/// longest axis with a 1-thick mid-plane separator, recurses, numbers
/// A, B, then the separator. Returns new-to-old.
fn geometric_nd_perm(m: usize) -> Vec<i32> {
    let idx = |x: usize, y: usize, z: usize| ((z * m + y) * m + x) as i32;
    let mut perm: Vec<i32> = Vec::with_capacity(m * m * m);
    // Box = [x0,x1) x [y0,y1) x [z0,z1). Emits vertex ids in
    // elimination order (leaves first, separators last).
    struct Rec<'a> {
        perm: &'a mut Vec<i32>,
        idx: &'a dyn Fn(usize, usize, usize) -> i32,
    }
    impl Rec<'_> {
        fn go(&mut self, x0: usize, x1: usize, y0: usize, y1: usize, z0: usize, z1: usize) {
            let (dx, dy, dz) = (x1 - x0, y1 - y0, z1 - z0);
            let n = dx * dy * dz;
            if n == 0 {
                return;
            }
            if n <= 2 || dx.max(dy).max(dz) <= 1 {
                for z in z0..z1 {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            self.perm.push((self.idx)(x, y, z));
                        }
                    }
                }
                return;
            }
            if dx >= dy && dx >= dz {
                let mid = x0 + dx / 2;
                self.go(x0, mid, y0, y1, z0, z1);
                self.go(mid + 1, x1, y0, y1, z0, z1);
                for z in z0..z1 {
                    for y in y0..y1 {
                        self.perm.push((self.idx)(mid, y, z));
                    }
                }
            } else if dy >= dz {
                let mid = y0 + dy / 2;
                self.go(x0, x1, y0, mid, z0, z1);
                self.go(x0, x1, mid + 1, y1, z0, z1);
                for z in z0..z1 {
                    for x in x0..x1 {
                        self.perm.push((self.idx)(x, mid, z));
                    }
                }
            } else {
                let mid = z0 + dz / 2;
                self.go(x0, x1, y0, y1, z0, mid);
                self.go(x0, x1, y0, y1, mid + 1, z1);
                for y in y0..y1 {
                    for x in x0..x1 {
                        self.perm.push((self.idx)(x, y, mid));
                    }
                }
            }
        }
    }
    Rec {
        perm: &mut perm,
        idx: &idx,
    }
    .go(0, m, 0, m, 0, m);
    perm
}

/// Geometric ND with AMD-ordered box leaves of at most `leaf_max`
/// vertices: perfect plane separators + the same leaf treatment as the
/// rslab-metis driver. Isolates separator quality from leaf handling.
fn geometric_nd_amd_perm(m: usize, leaf_max: usize) -> Vec<i32> {
    let idx = |x: usize, y: usize, z: usize| ((z * m + y) * m + x) as i32;
    let mut perm: Vec<i32> = Vec::with_capacity(m * m * m);
    struct Rec<'a> {
        perm: &'a mut Vec<i32>,
        idx: &'a dyn Fn(usize, usize, usize) -> i32,
        leaf_max: usize,
    }
    impl Rec<'_> {
        fn go(&mut self, x0: usize, x1: usize, y0: usize, y1: usize, z0: usize, z1: usize) {
            let (dx, dy, dz) = (x1 - x0, y1 - y0, z1 - z0);
            let n = dx * dy * dz;
            if n == 0 {
                return;
            }
            if n <= self.leaf_max || dx.max(dy).max(dz) <= 1 {
                // AMD on the leaf box (7-point pattern on dx x dy x dz).
                let lidx = |x: usize, y: usize, z: usize| ((z * dy + y) * dx + x) as i32;
                let mut cols: Vec<Vec<i32>> = vec![Vec::new(); n];
                for z in 0..dz {
                    for y in 0..dy {
                        for x in 0..dx {
                            let k = lidx(x, y, z) as usize;
                            cols[k].push(k as i32);
                            let mut push = |a: usize, b: i32| {
                                cols[a].push(b);
                                cols[b as usize].push(a as i32);
                            };
                            if x + 1 < dx {
                                push(k, lidx(x + 1, y, z));
                            }
                            if y + 1 < dy {
                                push(k, lidx(x, y + 1, z));
                            }
                            if z + 1 < dz {
                                push(k, lidx(x, y, z + 1));
                            }
                        }
                    }
                }
                let mut col_ptr: Vec<i32> = vec![0];
                let mut row_idx: Vec<i32> = Vec::new();
                for col in &mut cols {
                    col.sort_unstable();
                    row_idx.extend_from_slice(col);
                    col_ptr.push(row_idx.len() as i32);
                }
                let pat = CscPattern::new(n, &col_ptr, &row_idx).unwrap();
                let local = rslab_amd::amd_order(&pat).unwrap();
                for &l in &local {
                    let l = l as usize;
                    let (lz, rem) = (l / (dx * dy), l % (dx * dy));
                    let (ly, lx) = (rem / dx, rem % dx);
                    self.perm.push((self.idx)(x0 + lx, y0 + ly, z0 + lz));
                }
                return;
            }
            if dx >= dy && dx >= dz {
                let mid = x0 + dx / 2;
                self.go(x0, mid, y0, y1, z0, z1);
                self.go(mid + 1, x1, y0, y1, z0, z1);
                for z in z0..z1 {
                    for y in y0..y1 {
                        self.perm.push((self.idx)(mid, y, z));
                    }
                }
            } else if dy >= dz {
                let mid = y0 + dy / 2;
                self.go(x0, x1, y0, mid, z0, z1);
                self.go(x0, x1, mid + 1, y1, z0, z1);
                for z in z0..z1 {
                    for x in x0..x1 {
                        self.perm.push((self.idx)(x, mid, z));
                    }
                }
            } else {
                let mid = z0 + dz / 2;
                self.go(x0, x1, y0, y1, z0, mid);
                self.go(x0, x1, y0, y1, mid + 1, z1);
                for y in y0..y1 {
                    for x in x0..x1 {
                        self.perm.push((self.idx)(x, y, mid));
                    }
                }
            }
        }
    }
    Rec {
        perm: &mut perm,
        idx: &idx,
        leaf_max,
    }
    .go(0, m, 0, m, 0, m);
    perm
}

fn main() {
    let m: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let (n, cp, ri) = grid3d_pattern(m);
    eprintln!("grid {m}^3, n={n}, nnz={}", ri.len());
    let pat = CscPattern::new(n, &cp, &ri).unwrap();

    // AMD baseline in the same metric.
    let t0 = std::time::Instant::now();
    let amd_perm = rslab_amd::amd_order(&pat).unwrap();
    let (fill, flops) = fill_of_perm(n, &cp, &ri, &amd_perm);
    eprintln!(
        "{:<40} fill {:>10.3} M  flops {:>8.3e}  ({} ms)",
        "AMD baseline",
        fill as f64 / 1e6,
        flops,
        t0.elapsed().as_millis()
    );

    // Geometric ND oracle: the fill target for a perfect separator tree.
    let t0 = std::time::Instant::now();
    let geo_perm = geometric_nd_perm(m);
    assert_eq!(geo_perm.len(), n);
    let (fill, flops) = fill_of_perm(n, &cp, &ri, &geo_perm);
    eprintln!(
        "{:<40} fill {:>10.3} M  flops {:>8.3e}  ({} ms)",
        "geometric ND oracle",
        fill as f64 / 1e6,
        flops,
        t0.elapsed().as_millis()
    );

    for leaf in [30usize, 120, 200, 1000, 4000] {
        let t0 = std::time::Instant::now();
        let p = geometric_nd_amd_perm(m, leaf);
        assert_eq!(p.len(), n);
        let (fill, flops) = fill_of_perm(n, &cp, &ri, &p);
        eprintln!(
            "{:<40} fill {:>10.3} M  flops {:>8.3e}  ({} ms)",
            format!("geometric ND + AMD leaves <= {leaf}"),
            fill as f64 / 1e6,
            flops,
            t0.elapsed().as_millis()
        );
    }

    let run = |name: &str, opts: MetisOptions| {
        let t0 = std::time::Instant::now();
        match metis_order_full(&pat, &opts) {
            Ok((perm, _, stats)) => {
                let (fill, flops) = fill_of_perm(n, &cp, &ri, &perm);
                eprintln!(
                    "{:<40} fill {:>10.3} M  flops {:>8.3e}  sep {:>6}  leaves {:>4}  ({} ms)",
                    name,
                    fill as f64 / 1e6,
                    flops,
                    stats.n_separator_vertices,
                    stats.n_amd_leaf_calls,
                    t0.elapsed().as_millis()
                );
            }
            Err(e) => eprintln!("{name}: FAILED {e:?}"),
        }
    };

    let d = MetisOptions::default;
    run("metis default", d());
    run(
        "imbalance 0.05",
        MetisOptions {
            max_imbalance: 0.05,
            ..d()
        },
    );
    run(
        "imbalance 0.10",
        MetisOptions {
            max_imbalance: 0.10,
            ..d()
        },
    );
    run(
        "imbalance 0.35",
        MetisOptions {
            max_imbalance: 0.35,
            ..d()
        },
    );
    run(
        "fm_passes 20",
        MetisOptions {
            fm_passes: 20,
            ..d()
        },
    );
    run("niparts 15", MetisOptions { niparts: 15, ..d() });
    run(
        "amd_switch 60",
        MetisOptions {
            nd_to_amd_switch: 60,
            ..d()
        },
    );
    run(
        "amd_switch 400",
        MetisOptions {
            nd_to_amd_switch: 400,
            ..d()
        },
    );
    run(
        "coarsen_floor 40",
        MetisOptions {
            coarsen_floor: 40,
            ..d()
        },
    );
    run(
        "coarsen_floor 240",
        MetisOptions {
            coarsen_floor: 240,
            ..d()
        },
    );
    for seed in [2u64, 3, 4] {
        run(&format!("seed {seed}"), MetisOptions { seed, ..d() });
    }
}
