# rslab-kahip

KaHIP-style flow-based nested-dissection fill-reducing ordering for
sparse symmetric matrices (Sanders & Schulz 2011; data reduction per
Ost, Schulz & Strash 2021), implemented in pure Rust, clean-room
from the published papers.

- **License:** MIT
- **Dependencies:** `rslab-ordering-core`, `rslab-amd` (small-leaf
  base case), `rslab-metis` (multilevel coarsening plumbing).
- **MSRV:** stable Rust, edition 2021. No `unsafe`.

## What KaHIP does

KaHIP layers two ideas on top of the standard multilevel
nested-dissection skeleton:

1. **Data reduction (K1).** Before partitioning, eliminate vertices
   that don't affect the optimal separator: degree-1 vertices,
   degree-2 paths, twins, and neighborhood-subset vertices. The
   reductions are stored on a stack and replayed in reverse to lift
   the reduced-graph permutation back to original indices.
2. **Flow-based refinement (K3, K4).** At each uncoarsening level,
   refine the bisection by extracting a band around the cut and
   computing a maximum flow / minimum cut on the band's
   super-source / super-sink construction. The K4 phase converts an
   edge bisection into a node separator via König's-theorem
   reduction on the boundary-bipartite graph, producing narrower
   separators than greedy FM extraction.

The driver supports Fast / Eco / Strong modes (`KahipMode`) trading
quality for wall-clock time.

## Reference

- Sanders, P., and Schulz, C. (2011). *Engineering Multilevel Graph
  Partitioning Algorithms.* European Symposium on Algorithms (ESA).
  The kaffpa framework.
- Ost, L., Schulz, C., and Strash, D. (2021). *Engineering Data
  Reduction for Nested Dissection.* The K1 reduction rules used in
  the data-reduction pre-pass.

Full BibTeX in `dev/references.bib` of the parent repository.

## Contract

`rslab-kahip` conforms to the RSLAB ordering-crate contract defined
by [`rslab-ordering-core`](https://crates.io/crates/rslab-ordering-core).

```rust,ignore
use rslab_kahip::{kahip_order, CscPattern};

let col_ptr = [0, 2, 5, 8, 10];
let row_idx = [0, 1,  0, 1, 2,  1, 2, 3,  2, 3];
let pattern = CscPattern::new(4, &col_ptr, &row_idx).expect("valid CSC");
let perm = kahip_order(&pattern).expect("kahip_order");
```

## Clean-room status

Implemented from the published KaHIP / Ost-Schulz-Strash papers, not
from the MIT-licensed reference C++ codebase. Each phase (K1 data
reduction, K2 push-relabel max-flow, K3 flow-based edge refinement,
K4 boundary-bipartite node separator, K5 multilevel controller, K6
nested-dissection driver) is documented per-module in source
comments.

## License

MIT.
