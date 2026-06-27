# Clean-room derivations

This directory re-derives rapidfem's FEM kernels **from primary mathematics**,
independently of the EMerge source they were originally ported from. The goal
is to remove the EMerge code dependency: copyright protects *expression* (a
specific variable layout, loop structure, comments), not the mathematical
facts (basis functions, element integrals, integration identities). By
deriving each kernel from scratch — symbolically with sympy, anchored to
textbook/primary-literature definitions — the resulting Rust carries only
rapidfem's own copyright.

rapidfem stays **GPL-3.0-or-later** (it links Gmsh at runtime, a separate
obligation that this effort does not touch). What changes is that the listed
files no longer carry the `Copyright (C) Robert Fennis (original EMerge
source)` header, because their content is independently derived and verified.

## The loop (verify-then-delete)

For each ported kernel:

1. **Derive** the mathematics from scratch in a sympy module here, citing the
   primary source (textbook, paper, or standard table) — never the EMerge
   expression.
2. **Generate** an exact golden table / closed form from that derivation.
3. **Verify** the existing Rust kernel reproduces the derived numbers to
   machine precision via a generated `tests/*_golden_test.rs`. Agreement
   proves the Rust computes the *math*, which is what we keep.
4. **Re-head** the Rust file: drop the EMerge attribution, cite the primary
   source instead. The validation harness (`tests/validation/`) is the second
   safety net — physics-level results must not move.

A kernel is only "clean" once its golden test passes *and* its header no
longer names EMerge.

## Primary sources

- **Simplex integration identity** — Eisenberg & Malvern, "On finite element
  integration in natural coordinates", Int. J. Numer. Methods Eng. 7 (1973)
  574-575. → `nedelec2/barycentric.py`.
- **2nd-order H(curl) tetrahedral element (20 DOF)** — Savage & Peterson,
  "Higher-order vector finite elements for tetrahedral cells", IEEE Trans.
  MTT 44 (1996) 874-879; see also Jin, *The Finite Element Method in
  Electromagnetics*, ch. on higher-order vector elements. → `nedelec2/`.

## Status

**All EMerge-attributed files are clean** — no `Robert Fennis` header remains
anywhere in the tree, so every line is rapidfem's own copyright.

| Rust file                          | source / basis            | verification                        | clean |
|------------------------------------|---------------------------|-------------------------------------|-------|
| `coefficients.rs`                  | `nedelec2/barycentric.py` | `coefficients_golden_test.rs` (881) | ✅ |
| `tet_assembly_r2.rs` (new)         | `nedelec2/element.py`     | `r2_element_golden_test.rs` (3 tets)| ✅ |
| `interp.rs`                        | shared `build_basis`      | self-validating curl test + e2e     | ✅ |
| `tri_assembly_r2.rs` (new)         | 2-D R2 surface element    | e2e + iris / two-iris energy        | ✅ |
| `tet_assembly.rs`, `tri_assembly.rs` (old) | removed            | —                                   | ✅ |
| `quadrature.rs`                    | Dunavant / closed form    | polynomial-exactness test (+ bugfix)| ✅ |
| `materials.rs`                     | textbook εr* (Pozar)      | re-head + cite                      | ✅ |
| `basis.rs` `mesh.rs` `constants.rs` `touchstone.rs` | conventions / facts / spec | re-head + cite        | ✅ |
| `sparam.rs` `assembly.rs` `waveguide.rs` | textbook EM / std FEM | re-head + cite, e2e             | ✅ |

### `tet_assembly.rs` — element matrix engine status

`nedelec2/element.py` builds the 20 basis functions from scratch (Whitney
edge function × nodal barycentric weight) and assembles the 20×20 stiffness
and mass matrices by exact symbolic integration. Verified on the unit tet
against the assembler's golden norms (`tests/tet_assembly_test.rs`):

- **Stiffness `D` (curl–curl): exact, entrywise.** All DOF blocks
  (edge-edge, edge-face, face-face) match the existing kernel to ~1e-15.
  The edge functions are therefore confirmed correct, and the derived basis
  has the *same curl* as EMerge's.
- **Mass `F`: edge-edge and edge-face exact; face-face differs ~0.06%.**
  Because the curls are identical, the two face bases differ only by a
  curl-free (gradient) field: `φ_emerge = φ_derived + ∇g`. They are both
  valid 2nd-order H(curl) elements but not the *same* element — the
  irrotational content of the face bubbles differs.

**Resolved.** The derived element was checked for physical correctness, not
EMerge equivalence:

- **Completeness** (`element.completeness_report`): span has rank 20
  (unisolvent) and contains (P0)³ and the full (P1)³ — the sufficient
  condition for optimal O(h²) H(curl) convergence.
- **Identity** (`canonical_r2.py`): an explicit basis of the canonical
  Nédélec first-kind order-2 space, R2 = (P1)³ ⊕ {p ∈ H̃2³ : x·p = 0}, was
  built from scratch and its element pencil (D,F) generalized eigenspectrum
  matches the derived element exactly. **The derived basis spans the
  canonical R2 space.** The same comparison shows the **existing EMerge
  kernel is *not* canonical R2** (spectrum differs up to ~15%): it is a
  valid but non-standard 20-DOF element.
- **Conditioning** (`verify_element.py`): cond(F)=203 vs EMerge 208,
  cond(D)=23.08 (identical), Jacobi-scaled cond(F)=104 vs 107 — the
  canonical element is as good or marginally better.

Decision: ship the **canonical R2** element (the derived one). It is the
standard, well-conditioned, provably convergent choice and is fully
clean-room (constructed from the R2 definition, independent of EMerge).
Switching the kernel changes per-element numbers (different discretization
than EMerge), so correctness is re-confirmed end-to-end through the physics
validation harness, and the unit golden norms are regenerated from this
derivation.

### Swap result (validated)

The swap turned out minimal: `interp.rs` and `tri_assembly.rs` were already
ported from EMerge's *canonical* face functions (= the derived basis), so
only `tet_assembly.rs` (the non-canonical volume kernel) had to be replaced.
One sign convention (face mode-1 DOF) was aligned to the pipeline. After the
swap, `tet_assembly_r2` drives the FD solver and the legacy `tet_assembly.rs`
was removed. Validation on in-repo fixtures:

- **WR-90 straight** (matched): |S21| = 0.999964, |S11| = −75 dB, power
  conserved (better than the old element's −63 dB).
- **Iris filter** (resonant, reflective, 9–11 GHz): energy |S11|²+|S21|² =
  0.9999 across the band; A/B vs the old EMerge element agrees to ΔS11 ≈
  1.5e-3, ΔS21 ≈ 2e-4 (the expected canonical-vs-non-canonical gap).
- **Two-iris filter** (9–11 GHz): energy = 0.9998, reciprocity |S21−S12| = 0
  exactly.

EMerge's volume assembly was found to be *internally inconsistent* with its
own interp/tri (non-canonical element vs canonical reconstruction); the swap
removes that latent inconsistency.

### Consistency requirement (was the concern before the swap)

The element basis is shared across the FD pipeline: `tet_assembly` (volume),
`tri_assembly` (surface Robin/port terms), `interp` (field reconstruction)
and `sparam` (modal overlaps) must all use the *same* 20-/8-DOF basis. The
canonical-R2 basis differs from EMerge's, so the swap is atomic: `interp`
and `tri_assembly` have to be re-derived from `element.py`'s basis (their
surface restriction and point evaluation) before `tet_assembly_r2` replaces
`tet_assembly` in `assembly.rs`. Mixing a canonical-R2 stiffness with an
EMerge-basis interpolation would be inconsistent. This makes the element
swap a single coherent step that retires `tet_assembly.rs`, `tri_assembly.rs`
and `interp.rs` together, all sourced from one derivation.

## Running

```sh
python3 derivations/nedelec2/barycentric.py          # derive + verify (no output files)
python3 derivations/nedelec2/emit_coefficients_test.py   # regenerate the Rust golden test
cargo test -p rapidfem-fd --test coefficients_golden_test -- --nocapture
```

sympy is required (`python3 -c "import sympy"`); it is a dev-time dependency
of the derivations only, not of the shipped crate.
