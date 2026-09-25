# rslab-metis

Multilevel nested-dissection fill-reducing ordering for sparse
symmetric matrices (Karypis & Kumar 1998), implemented in pure Rust,
clean-room from the published METIS papers.

- **License:** MIT
- **Dependencies:** `rslab-ordering-core`, `rslab-amd` (used as the
  small-leaf base case), `rayon` (the two sides of a bisection are
  ordered in parallel).
- **MSRV:** stable Rust, edition 2021. No `unsafe`.

## What METIS does

METIS computes a fill-reducing permutation by recursively bisecting
the adjacency graph of `A`. At each level it (1) coarsens the graph
by heavy-edge matching, (2) computes an initial edge bisection on the
coarsest graph, refined with Fiduccia-Mattheyses (FM), (3) turns it
into a vertex separator there, and (4) uncoarsens, refining the
vertex separator at every level. The two halves recurse
independently; small subgraphs fall through to AMD as the leaf
algorithm. The resulting permutation tends to outperform AMD on
large 2D / 3D mesh-shaped problems and underperform AMD on small or
arrow-structured matrices - picking the right backend is a
problem-dependent choice that downstream solvers (and RSLAB itself)
make at the analysis boundary.

## Reference

- Karypis, G., and Kumar, V. (1998). *A Fast and Highly Quality
  Multilevel Scheme for Partitioning Irregular Graphs.* SIAM Journal
  on Scientific Computing, 20(1), 359-392. The original METIS paper
  describing multilevel coarsening, initial partitioning, and FM
  refinement.
- Karypis, G., and Kumar, V. (1999). *A Fast and Highly Quality
  Multilevel Scheme for Partitioning Irregular Graphs.* Companion
  paper covering the nested-dissection driver.

## Contract

`rslab-metis` conforms to the RSLAB ordering-crate contract defined
by [`rslab-ordering-core`](https://crates.io/crates/rslab-ordering-core).

```rust,ignore
use rslab_metis::{metis_order, CscPattern};

let col_ptr = [0, 2, 5, 8, 10];
let row_idx = [0, 1,  0, 1, 2,  1, 2, 3,  2, 3];
let pattern = CscPattern::new(4, &col_ptr, &row_idx).expect("valid CSC");
let perm = metis_order(&pattern).expect("metis_order");
```

## Clean-room status

Implemented from the published METIS papers, not from the
GPL-licensed reference C codebase. Algorithmic decisions (matching
heuristic, initial-partition strategy, FM gain computation, separator
refinement, leaf threshold) are documented per-module in source
comments.

## License

MIT.
