# Vendored rslab

Library copy of the rslab sparse direct solver
(https://github.com/milanofthe/rslab).

- Vendored from: commit `08f06fa` (rslab 0.38.0), "Test race, LU matching and KLU thread invariance, race settings and complex KLU export; fix stale diagnostics docs (#134)"
- Contents: `src/` (library only) plus the ordering crates
  `crates/{rslab-ordering-core,rslab-amd,rslab-amf,rslab-metis}`, the license files, and a manifest trimmed
  to the library.
- Local changes: none. Do not patch this tree; fix upstream and resync.

## Resync

From a checkout of rslab next to this repository:

```sh
python ../rslab/tools/vendor.py vendor/rslab --rev <commit>
```

then build and test this repository (cargo updates `Cargo.lock`).
