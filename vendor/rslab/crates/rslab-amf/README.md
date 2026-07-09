# rslab-amf

Approximate Minimum Fill (AMF / HAMF4) fill-reducing ordering for
sparse symmetric matrices, implemented in pure Rust on the
quotient-graph framework of Amestoy (1999).

- **License:** MIT
- **Dependencies:** `rslab-ordering-core` only.
- **MSRV:** stable Rust, edition 2021. No `unsafe`.

## What AMF does

AMF is a quotient-graph greedy ordering closely related to AMD
(`rslab-amd`), but the per-pivot scoring metric is the *approximate
fill* introduced by selecting that pivot rather than the
*approximate degree*. Empirically this produces lower-fill
permutations than AMD on a substantial fraction of structurally
unsymmetric or arrow-shaped matrices, at modest extra cost per
pivot. The HAMF4 variant inherits AMD's mass elimination,
supervariable detection, aggressive absorption, and dense-row
deferral.

## Reference

- Amestoy, P. R. (1999). *Recent progress in parallel multifrontal
  solvers.* Proceedings of the Fifteenth World Congress on
  Scientific Computation, Modelling and Applied Mathematics
  (IMACS-15). The HAMF / HAMF4 algorithm.

Full BibTeX in `dev/references.bib` of the parent repository.

## Contract

`rslab-amf` conforms to the RSLAB ordering-crate contract defined by
[`rslab-ordering-core`](https://crates.io/crates/rslab-ordering-core).
Input is a borrowed full-symmetric `CscPattern<'_>`; output is a
`Vec<i32>` permutation plus shared and crate-specific stats.

```rust,ignore
use rslab_amf::{amf_order, CscPattern};

let col_ptr = [0, 2, 5, 8, 10];
let row_idx = [0, 1,  0, 1, 2,  1, 2, 3,  2, 3];
let pattern = CscPattern::new(4, &col_ptr, &row_idx).expect("valid CSC");
let perm = amf_order(&pattern).expect("amf_order");
```

See `rslab-amd`'s README for a fuller worked example; the API shape
is identical.

## License

MIT.
