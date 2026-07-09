# rslab-scotch

SCOTCH-style nested-dissection fill-reducing ordering for sparse
symmetric matrices (Pellegrini 1996), implemented in pure Rust,
clean-room from the published SCOTCH papers.

- **License:** MIT
- **Dependencies:** `rslab-ordering-core`, `rslab-amd` (small-leaf
  base case), `rslab-metis` (multilevel coarsening plumbing).
- **MSRV:** stable Rust, edition 2021. No `unsafe`.

## What SCOTCH does

SCOTCH and METIS share the multilevel-recursive-bisection skeleton
but differ in the bisection refinement strategy and the separator
extraction. SCOTCH uses banded FM and halo refinement to produce
narrower vertex separators than METIS on graphs with strong
locality, particularly finite-element meshes with high aspect
ratio. `rslab-scotch` reuses the matching / coarsening /
initial-partition machinery from `rslab-metis` and substitutes
SCOTCH-style refinement on top.

## Reference

- Pellegrini, F. (1996). *Application of graph partitioning
  techniques to static mapping and domain decomposition.* PhD
  thesis, Université Bordeaux 1. The original SCOTCH framework.
- Pellegrini, F., and Roman, J. (1996). *SCOTCH: A software package
  for static mapping by dual recursive bipartitioning of process and
  architecture graphs.* Lecture Notes in Computer Science 1067,
  493-498.

Full BibTeX in `dev/references.bib` of the parent repository.

## Contract

`rslab-scotch` conforms to the RSLAB ordering-crate contract defined
by [`rslab-ordering-core`](https://crates.io/crates/rslab-ordering-core).

```rust,ignore
use rslab_scotch::{scotch_order, CscPattern};

let col_ptr = [0, 2, 5, 8, 10];
let row_idx = [0, 1,  0, 1, 2,  1, 2, 3,  2, 3];
let pattern = CscPattern::new(4, &col_ptr, &row_idx).expect("valid CSC");
let perm = scotch_order(&pattern).expect("scotch_order");
```

## Clean-room status

Implemented from the published SCOTCH papers, not from the
CeCILL-C licensed reference C codebase. The bisection-refinement
algorithms (band FM, halo FM, vertex-separator extraction) are
documented per-module in source comments.

## License

MIT.
