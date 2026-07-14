# RapidFEM — Frequency-Domain Element Basis — Implementation Plan

Modular curl-conforming elements for `rapidfem-fd`: pluggable basis, and a
mixed-order (`hp`) space in a single problem.

## Goal

Today `rapidfem-fd` hard-codes exactly one element: the canonical Nédélec
first-kind order-2 space, in an **interpolatory** basis, at 20 DOFs per
tetrahedron, everywhere in the mesh. The element sizes are baked into Rust
*types* (`[usize; 20]`, `[[C64; 8]; 8]`, `chunks_mut(400)`), so there is no seam
at which a different element could be substituted, and no way to give one
tetrahedron a different order from its neighbour.

Three things follow that we want and cannot have:

1. **Order 1 is not reachable.** The current basis is not hierarchical:
   `span{L_a·W_ab, L_b·W_ab}` does not contain `W_ab`, because `L_a + L_b ≠ 1`
   away from the edge. (Verified symbolically: none of the six Whitney functions
   lies in the mode-1 block.) So a lowest-order solve cannot be had by masking
   DOFs; it needs its own element.

2. **Mixed order is impossible.** An order-1 and an order-2 element sharing an
   edge must agree on the tangential trace along it. With an interpolatory basis
   their traces live in different spaces, so conformity fails and would need
   constraint equations.

3. **Conditioning is left on the table.** The measured `cond(D+F) = 51.1` on a
   regular tetrahedron drops to `25.7` under an orientation-aware hierarchical
   basis (`derivations/basis_nondim/orthonormal_basis.py`).

The payoff is DOF economy. A mesh is refined for two quite different reasons and
only one of them calls for a high order:

- Refined because the **wavelength** demands it → `kh = O(1)` → numerical
  dispersion dominates → order 2 earns its DOFs (the phase error per wavelength
  scales as `(kh)^{2p}`).
- Refined because the **geometry** demands it (a sub-micron trace, a via fabric,
  a fillet) → `kh ≪ 1` already → the dispersion error is negligible and the
  order-2 enrichment buys accuracy nothing needs. These elements can run at
  order 1 and drop from ≈6.3 to ≈1.2 DOFs each.

A second, independent reason points the same way: at a re-entrant PEC edge the
exact field has an algebraic singularity `|E| ~ r^(ν−1)`, `ν < 1`, and polynomial
order does not converge against a singularity — the rate is set by `ν`, not by
`p`. High order in the smooth far field, low order and small `h` at the
singularity, is the classical `hp` strategy, and it is precisely the pattern an
on-chip or a filleted-horn mesh already has.

Background, the family/order comparison table, and the convergence rates are in
`docs/report/` (§ "Which element, and why" and § "A hierarchical basis, and
variable order").

## Guiding principles

1. **Refactor first, build second.** Stages 0–2 restructure the element
   machinery with *zero behaviour change*: the golden tests must pass
   bit-identically and the physics fixtures must not move. Only then is a new
   basis added. Never debug a refactor and a new element at the same time.

2. **Every stage has an oracle.** Stages 0–2 are bit-identity. Stage 3 has a
   stronger one: two bases of the *same* space produce the *same* discrete
   solution in exact arithmetic, so the swap must leave the S-parameters
   unchanged to solver tolerance. Stage 4 must reproduce stage 3 exactly at
   uniform order.

3. **The assembly must not know which element it is assembling.** The seam is a
   term list, not a trait object threaded through the hot loop.

4. **Derive, do not transcribe.** Every new element gets a sympy derivation under
   `derivations/` that emits its golden test, exactly as `nedelec2/` does today.

## The conformity rule

A conforming mixed-order H(curl) space needs nothing global. The rule is local:

> Every geometric entity (edge, face) carries an order `p_E`. Every element
> incident on that entity uses exactly the DOFs that entity owns. With a
> **hierarchical** basis those are a prefix of both elements' bases, so the
> tangential traces agree automatically — no constraint equations.

The element order is assigned per cell; the entity orders follow by the
**minimum rule**:

```
p_edge[e] = min over tets containing e   of p_cell[t]
p_face[f] = min over tets containing f   of p_cell[t]     (1 or 2 tets)
p_cell[t] = the cell's own order                          (interior DOFs, p ≥ 3)
```

A face between a `p=1` and a `p=2` cell therefore carries zero face DOFs, and its
edges one each.

### The caveat that decides which hierarchical basis

Not every hierarchical basis is admissible. The construction must preserve the
**local exact sequence** (the discrete gradients must lie exactly in the space),
or the spurious modes that edge elements exist to eliminate come back. This is
the content of Schöberl & Zaglmayr, *High order Nédélec elements with local
complete sequence properties*, COMPEL 24 (2005) 374–384 — the title is the
requirement. We take that construction, not an ad-hoc nesting.

## Where the current code is welded shut

| File | Line | What |
|---|---|---|
| `basis.rs` | 36, 38, 40 | `tet_to_field: Vec<[usize; 20]>`, `tri_to_field: Vec<[usize; 8]>`, `edge_to_field: Vec<[usize; 2]>` — DOF counts in the *type* |
| `basis.rs` | 52 | `n_field = 2·n_edges + 2·n_tris` — fixed stride |
| `basis.rs` | 141 | `empty_tri_matrix` → `n_tris · 64` |
| `tet_assembly_r2.rs` | 143–156 | `Term { mono: [usize; 2] }`, `BasisFn { terms: [Term; 2] }` — exactly two terms, degree-2 monomial implied |
| `tet_assembly_r2.rs` | 245 | `r2_tet_stiff_mass(...) -> ([[C64; 20]; 20], [[C64; 20]; 20])` |
| `tet_assembly_r2.rs` | 322, 329 | `nnz = n_tets · 400`, `chunks_mut(400)` — the lock-free parallel scatter rests on the constant chunk size |
| `tri_assembly_r2.rs` | 31, 97–132 | its **own** `Term`/`SurfFn`, its own `TRI_EDGE_MAP`, its own `build_surface_basis(...) -> [SurfFn; 8]` |
| `assembly.rs` | 155, 235 | `ti · 64` into the flat surface buffer |
| `coefficients.rs` | 28–41 | `volume_coeff(a,b,c,d)` takes *vertex indices*, not exponents; `FACTORIALS` has 10 entries |

Consumers of the basis, all of which must follow any change:

- `tet_assembly_r2::assemble_global_matrices` (volume `E`, `B`)
- `tri_assembly_r2::{ned2_tri_stiff, ned2_tri_force}` (Robin term, load vector)
- `interp::{eval_field_in_tet, eval_curl_in_tet}` (field reconstruction), and
  through it `sparam`, `farfield`, `error_estimator`, `vtk_export`, `simulation`
- `eigenmode` (via `assemble_global_matrices`)

## Two findings that make this cheaper than it looks

### 1. The general term representation already exists — in the derivation

`derivations/nedelec2/element.py` represents a vector field as a list of
`(coeff, exps, vec)` with `exps` a 4-tuple of barycentric exponents. That is the
*general* form for any polynomial H(curl) basis. The Rust specialised it to
"exactly two terms, exponents implied by `mono: [usize; 2]`".

Generalising Rust is therefore not a new design — it is **porting the derivation
back**:

```rust
/// c · L1^e1 L2^e2 L3^e3 L4^e4 · ∇L_g
pub struct Term { pub coeff: f64, pub exps: [u8; 4], pub grad: u8 }
```

The exact integration survives untouched, because the closed form takes
exponents, not indices:

```
∫_T L^α dV / 6V = (∏ α_i!) / (Σ α_i + 3)!
```

The existing `FACTORIALS` table runs to `9!`, which already covers order 3 (two
degree-3 functions → degree 6, `+3 = 9`). Order 4 needs `11!`; a one-line
extension.

The curl of a term is up to four terms of the shape `c · L^β · v` with `v` a
*constant vector* (not a `∇L`), so the curl needs its own term type — which is
exactly what `element.py::curl_field` already produces.

### 2. The surface element is the trace of the volume element

Verified symbolically on a generic (sheared) tetrahedron
(`derivations/nedelec2/` — to be added as `face_trace.py`):

- For a vertex `o` **not** on a face `f`, the tangential component of `∇L_o`
  vanishes **identically** on `f`. (Both `Σ_i ∇L_i = 0` in 3-D and
  `Σ_{i∈f} ∇₂L_i = 0` on the triangle.)
- Hence every volume term whose gradient is `∇L_o` contributes nothing to the
  trace, and every term carrying a positive power of `L_o` vanishes because
  `L_o = 0` there. **The trace is a purely mechanical restriction of the term
  list.**
- For `i ∈ f`, the tangential projection of the 3-D `∇L_i` *is* the 2-D
  barycentric gradient of the face triangle, exactly.
- Measured: the traced span has rank 8, and exactly 8 volume DOFs have a nonzero
  trace — the face's 3 edges × 2 modes, plus that face × 2 modes.

So `tri_assembly_r2::build_surface_basis` is a **redundant second
construction**. It is also the one carrying the sign-convention hazard: line 118
says "sign-matched to volume", and `interp.rs:31` carries a `RECON_SIGN = -1.0`
to reconcile the reconstruction with the excitation convention. Deriving the
surface element as the trace removes that class of bug **structurally**, not by
discipline.

## Target architecture

The seam is a term list. The expensive `O(n²)` work is basis-agnostic.

```rust
/// Which geometric entity a local DOF belongs to, and its index within it.
pub enum DofOwner { Edge(u8, u8), Face(u8, u8), Cell(u8) }

/// The element, evaluated for one tetrahedron and one order signature.
pub struct ElementBasis {
    pub funcs: Vec<SmallVec<[Term; 4]>>,   // per local DOF
    pub owner: Vec<DofOwner>,              // same length
}

/// Per-cell order signature, derived from the cell orders by the minimum rule.
pub struct OrderSig { pub cell: u8, pub edge: [u8; 6], pub face: [u8; 4] }

pub trait CurlBasis: Send + Sync {
    fn build(&self, geom: &TetGeom, sig: &OrderSig) -> ElementBasis;
    /// DOF count an entity of this kind owns at a given order.
    fn n_dofs_on(&self, kind: EntityKind, order: u8) -> usize;
}

/// Basis-agnostic. All the O(n²) work lives here.
pub fn stiff_mass(
    b: &ElementBasis, grads: &[V3; 4], six_v: f64,
    eps: &Tensor3, mu_inv: &Tensor3,
) -> (Vec<C64>, Vec<C64>);   // row-major, n×n
```

The basis is built once per tetrahedron (cheap); the quadratic work is in
`stiff_mass`, which never sees the trait. So `&dyn CurlBasis` costs nothing
measurable and we avoid threading generics through the whole stack.

### DOF map

Fixed stride → prefix-sum offsets, one entry per geometric entity:

```rust
pub struct DofMap {
    edge_off: Vec<u32>,   // len n_edges + 1
    face_off: Vec<u32>,   // len n_faces + 1
    cell_off: Vec<u32>,   // len n_tets  + 1
    edge_base: usize, face_base: usize, cell_base: usize,
    pub n_field: usize,
}
```

Global index of the `k`-th DOF of edge `e` is `edge_base + edge_off[e] + k`.

### Assembly

`chunks_mut(400)` → a prefix sum over the per-element entry counts `n_i²`, then
`split_at_mut` in a loop to hand each rayon worker a disjoint slice. The
lock-free property is preserved; only the chunk boundaries become data instead of
a constant.

## Stages

Each stage is independently shippable and has its own oracle.

### Status

Stages 0-3 are done, on `feat/fd-modular-element`. What actually landed:

| Stage | Commit | Outcome |
|---|---|---|
| oracle | `c5bd7d0` | `tests/global_assembly_pin_test.rs`: DOF map, scatter order, matrix values |
| 0 | `48e57c6` | bit-identical, and **15% faster** (31.3 ms vs 36.7 ms on 10 368 tets) |
| 1 | `1815b40` | values bit-identical; the DOF numbering is a proven permutation of the old |
| 2 | `c792f4f` | surface element built from the volume element's generators; trace identity proved |
| 3 | `5c809fd` | hierarchical basis; same space to 1e-14 across the whole cavity spectrum |

Two things came out differently from the plan.

**Stage 0 got faster, not slower.** The general term representation made it natural
to hoist the curls out of the `(i,j)` double loop and compute them once per basis
function. The old code recomputed both curls inside the loop, so it did O(n²) cross
products where O(n) suffice.

**Stage 1 changes the DOF numbering.** The old layout was mode-major
(`[all edges m1][all faces m1][all edges m2][all faces m2]`), which only exists if
every entity has the same count — the exact assumption being removed. The new one is
entity-major, so an entity's DOFs are contiguous and its count is free. That is a
relabelling of the unknowns, not a change of the discretisation: the assembled
system is `P·K·Pᵀ`. `numbering_is_a_relabelling_of_the_mode_major_layout` proves the
permutation is a bijection, and the pin's abs-sum / Frobenius values (invariant under
a permutation) did not move a single digit. Only the pattern hash did.

**Stage 2 did not need the adjacent tetrahedron.** The plan proposed restricting the
volume element of the neighbouring tet onto the face at runtime. That is unnecessary:
the trace of the volume function on a face is the *same formula* read with 2-D
gradients (`derivations/nedelec2/face_trace.py`, lemmas L1-L3), so it is enough for
the surface element to call the volume element's own generators (`r2_edge_fns`,
`r2_face_fns`) on the triangle's three nodes. The surface path stays tet-free and
cheap, and the signs cannot drift because they are no longer written down twice.
`tests/face_trace_test.rs` closes the loop numerically: it integrates the traced
volume functions over the face and gets `ned2_tri_stiff` back, using the DOF
correspondence the *assembler* uses (a shared global index) rather than a
hand-derived one. That correspondence turns out to be permuted — the triangle's edges
map to tet-local edges `[0, 3, 1]`, not `[0, 1, 3]` — which is precisely the kind of
thing the old "sign-matched to volume" comment was silently carrying.

Removed on the way: `AreaCoeffCache` and `VolumeCoeffCache` (625-entry tables built
per solve, no longer read by anything), and the `ac_base` parameter of
`ned2_tri_stiff`. The regenerated `tri_mass_golden_test` values are byte-identical.

**Stage 3's conditioning claim was wrong.** The plan said to expect `cond(D+F)`
51.1 → 25.7 on a regular tet. Measured (`hierarchical.py`, P5): 567 → 499 on the
unit tet (1.14× better), 527 → **727** on a general tet (*worse*), 570 772 → 376 115
on a sliver (1.52× better). The hierarchical basis is not a conditioning win in
general. Its reason to exist is the nesting, and that is what the measurements do
confirm: mode 0 of the edges *is* the Whitney space (rank 6, identical), whereas the
interpolatory mode-0 block is disjoint from it (rank 12 = 6 + 6) and contains no
Whitney function at all. Without that, stage 4 is impossible.

**A pre-existing defect surfaced while building stage 3's oracle.**
`eigenmode::solve_eigenmode` runs its Lanczos recurrence in the Euclidean inner
product on `(E − σB)⁻¹B`, which is self-adjoint only in the `B` inner product, with
local-only reorthogonalisation and no residual check. Its eigenvalues land near the
truth, but every returned eigenVECTOR has an O(1) eigenpair residual, and it reports
ghost modes below the fundamental (a cluster near 5.3 GHz in a cavity whose true
spectrum, computed densely, has nothing between 0 and 8.245 GHz). This is unrelated
to the element work and is to be fixed separately. It is documented at the top of
`eigenmode.rs`; the stage-3 oracle deliberately does not use it.

### Stage 0 — general term representation (pure refactor)

- `Term { coeff, exps: [u8; 4], grad: u8 }`; `BasisFn.terms: SmallVec<[Term; 4]>`.
- `coefficients::volume_coeff` / `area_coeff` take **exponents**, keep a thin
  index-taking wrapper for the existing golden test.
- `r2_tet_stiff_mass` loops over `basis.len()` and over the term list; returns
  `Vec<C64>` (row-major) instead of `[[C64; 20]; 20]`.
- `interp` follows (it already reads `build_basis`).
- R2 itself is **unchanged**.

**Oracle:** every golden test passes bit-identically
(`r2_element_golden_test`, `coefficients_golden_test`, `interp_golden_test`,
`tri_mass_golden_test`), and the WR-90 / iris fixtures produce bit-identical
S-parameters.

### Stage 1 — variable-size DOF map and assembly

- `Nedelec2Basis` → `DofMap` with entity offsets.
- Assembly: prefix-sum offsets, `split_at_mut` scatter.
- Surface buffer: `ti · 64` → offset table.
- Orders still uniform at 2, so every count comes out as before.

**Oracle:** bit-identical, as stage 0.

### Stage 2 — surface element as the volume trace

- New `derivations/nedelec2/face_trace.py`: prove the two trace facts, emit the
  golden.
- `tri_assembly_r2::build_surface_basis` **deleted**. The Robin `8×8` and the
  load vector are computed by restricting the volume `ElementBasis` of the
  adjacent tetrahedron to the face.
- `TRI_EDGE_MAP` and the "sign-matched to volume" comment go with it.

**Oracle:** the traced Robin matrix must equal the current one entrywise (this
also *proves* the present sign convention, rather than asserting it). Fixtures
bit-identical.

### Stage 3 — hierarchical R2 basis

- New `derivations/nedelec2/hierarchical.py`: the Schöberl–Zaglmayr construction
  (integrated Legendre), with the local exact sequence checked explicitly (the
  discrete gradients must be *in* the space, not merely near it).
- Verify it spans the same `R2` by the generalised-eigenspectrum test already
  used in `canonical_r2.py`.
- Register it as a second `CurlBasis`. Select via config; interpolatory stays the
  default until stage 3 is validated.

**Oracle (strong):** two bases of the same space give the same discrete solution
in exact arithmetic. So the S-parameters must agree with the interpolatory basis
to **solver tolerance** (not bit-identical — different roundoff). If they do not,
the two bases do not span the same space and the derivation is wrong.

**Expected side effect:** `cond(D+F)` 51.1 → 25.7 on a regular tet; measurable as
fewer perturbed pivots and a smaller residual on the sliver-adjacent fixtures.

### Stage 4 — variable order

- `OrderSig` per cell; minimum rule for entities.
- `DofMap` counts per entity from `n_dofs_on(kind, p_E)`.
- Element order 1 becomes the mode-0 block, free.

**Oracle:** at uniform `p = 2` the result must be **bit-identical** to stage 3.
At uniform `p = 1` it must reproduce a separately derived Whitney element.
Convergence study: `p = 1` must show `O(h)` and `p = 2` `O(h²)` in the energy
norm on a manufactured solution.

### Stage 5 — order policy

A-priori, needs no solve:

```
p_K = 1   if  k_K · h_K < θ      (geometry-driven)
p_K = 2   otherwise              (wavelength-driven)
```

with `k_K = ω·√(εr·μr)/c₀` from the **element's own material** and `h_K` the
element *diameter* (the right measure: a cell that is thin transversally but long
axially is correctly flagged wavelength-driven).

Later: a p-decay indicator, which is **free** with a hierarchical basis — the
magnitude of the top-mode coefficients on an element. Small ⇒ smooth ⇒ raise `p`;
large ⇒ singular ⇒ refine `h`. That is exactly the decision an `hp` strategy has
to make, and without a hierarchy there is no way to make it.

**Oracle:** the sweep with the policy active must match uniform `p = 2` to within
the accuracy target on the fixtures, at a measured DOF and solve-time reduction.

## Risks and open questions

- **Exact sequence.** The one way to get this badly wrong is a hierarchical basis
  that breaks the discrete de Rham complex, reintroducing spurious modes. Stage 3
  must check it explicitly, not assume it.
- **Hot-path cost of the general representation.** `[u8; 4]` exponents and a
  `SmallVec` term list versus today's two fixed terms. Expected comparable;
  measure with `examples/solver_bench.rs` at stage 0 and do not proceed if the
  element assembly regresses materially.
- **Conditioning under mixed order.** Mixing orders widens the diagonal scale
  spread. The symmetric equilibration already in place should absorb it; verify
  rather than assume.
- **Static condensation** only pays from `p ≥ 3` (no cell-interior DOFs below
  that). Out of scope until an order-3 element exists.
- **`RECON_SIGN`** in `interp.rs` is a global reconstruction sign, independent of
  the surface/volume duplication. Stage 2 should establish whether it is still
  needed once the trace is derived rather than mirrored.

## Not in scope

- Isoparametric (curved) elements. They would break the exact integration (the
  Jacobian stops being constant) and bring quadrature back into the element. They
  are also the prerequisite for order ≥ 3 to pay on a curved boundary — see the
  report's § "The order: why not 3".
- The time-domain backend. Its nodal DG elements are decoupled and already take
  the order as a constructor argument; the problem does not arise there in the
  same form.
