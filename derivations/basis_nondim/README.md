# Better basis (①) & non-dimensionalization (④) — derivation & findings

Two levers for the canonical R2 formulation, derived against the *exact*
element rapidfem assembles (`derivations/nedelec2/element.py`) and the real
global pipeline (`crates/rapidfem-fd/src/assembly.rs`: `K = E − k0²B`, plus the
per-DOF diagonal equilibration already in place). Run:

```sh
python3 derivations/basis_nondim/orthonormal_basis.py   # lever ①
python3 derivations/basis_nondim/nondimensionalize.py   # lever ④
```

Both are *solution-preserving congruences* — they change the basis/units, never
the discrete physics.

---

## ① Better-conditioned basis — real but modest, needs a hierarchical basis

**The assembly-consistency constraint.** A local basis change `φ' = Tφ` is a
global congruence `K → PᵀKP` only if every element sharing a DOF applies the
*same* transform to it. At order 2 there are no cell-interior DOFs — edge DOFs
are shared by all tets around the edge, face DOFs by the two across the face —
so `T` must be **block-diagonal by geometric entity** (a 2×2 on each edge pair,
a 2×2 on each face pair). The dense whitening `T=(D+F)^{-1/2}` that would give
`cond=1` mixes edge with face DOFs and is **not assemblable** — only a lower
bound.

**What the numbers say** (regular tet = best case, so cond > 1 is pure basis
penalty, not geometry):

| stage | cond(D+F), regular tet |
|---|---|
| raw canonical basis | 71.6 |
| per-DOF equilibration **(current)** | 51.1 |
| + admissible entity-block 2×2 **(lever ①)** | **25.7** |
| dense `(D+F)^{-1/2}` (ideal, *not* assemblable) | 1.0 |

So the basis lever buys a consistent **~2× over the equilibration we already
do** (also ~1.7× on a distorted tet, ~1.5× on a sliver). Note the regular-tet
baseline (~51) is already benign — the old "600" was the unit *right*-tet,
itself distorted.

**Why it is not a cheap constant.** The within-entity coupling has
near-constant magnitude (`|ρ| ≈ 0.495`) but its **sign flips with the entity's
orientation** (edges `+0.49`, faces `−0.50` on the regular tet). A sign-blind
constant 2×2 therefore *re-correlates* half the blocks and makes conditioning
**worse** (51 → 95). An **orientation-aware** decorrelator (constant `|ρ|`,
per-entity sign from the canonical edge/face ordering the mesh already
provides) recovers the per-element ceiling (51 → 25.7, matching the 25.68
ideal-entity-block value) and is exactly solution-preserving (`x = Px'` to
1e-15). That orientation-aware construction *is* a hierarchical
(integrated-Legendre / Schöberl–Zaglmayr) basis — i.e. lever ① done properly is
lever ②, a rewrite of `basis.rs` + the element, not a patch.

**Verdict ①.** A ~2× element-conditioning gain, geometrically irrelevant to
slivers, achievable only via an orientation-aware hierarchical basis. Low
priority: the global system conditioning is dominated by mesh scale/frequency
(④) and geometry (own-mesher), not by this constant factor. Park it with ②.

---

## ④ Non-dimensionalization — the RFIC lever (unit-robustness, not conditioning)

**The derivation.** Scale coordinates `x = L0·x̃`. The canonical basis is
scale-invariant (`φ ~ O(1)`), while `curl φ = (1/L0)curl̃ φ̃` and `dV = L0³dṼ`,
so with relative material tensors:

```
E = L0 · Ẽ ,   B = L0³ · B̃ ,   K = E − k0²B = L0 · [ Ẽ − (k0 L0)² B̃ ]
```

verified exact (1). The whole system is the dimensionless block `Ẽ − κ²B̃`
times a scalar `L0`, governed by one number **κ = k0·L0** (the electrical size
of `L0`). For RFIC (features ~1 µm, f ~ 60 GHz) `κ ≈ 1.3e-3`, so the mass term
is a ~1e-6 perturbation of the stiffness — physically correct (electrically
tiny).

**Honest scope: ④ does NOT change conditioning.** Uniform scaling is a
similarity: `K_raw = L0·K̃`, so `cond(K_raw) = cond(K̃)` to machine precision —
verified across 6 orders of `L0` at fixed physics (A). Equilibration already
removes the uniform scale from the *solve*. So ④ is **not** a conditioning fix.

**What ④ actually buys: unit-robustness of every absolute tolerance.**
rapidfem's tolerances (`SINGULAR_EPS=1e-30`, the sliver floor, solver pivot
drops, `LANCZOS_BREAKDOWN`) are compared against entry magnitudes that scale as
`L0^p`:

- A **geometrically perfect** regular tet of 10 pm edge has `6V ≈ 7e-34 <
  SINGULAR_EPS` and is **wrongly declared degenerate** (B) — purely a unit
  artifact; in `x̃` its `6Ṽ = O(1)` always.
- The *same physical problem* meshed in m / mm / µm produces raw `K` spanning
  `10^9` in magnitude but a **byte-identical** `K̃` (B′) — only the
  threshold decisions move.

`L0 = ` geometric-mean edge length centers entries at `O(1)`, maximally far
from underflow and from the absolute tolerances (5).

**Verdict ④.** Cheap, global, solution-preserving, and the **enabling fix for
sub-micron RFIC**: assemble in `x̃ = x/L0` (L0 = mean mesh edge length), carry
`κ = k0 L0`, solve, rescale the reconstructed field back by `L0`. S-parameters
are ratios → already dimensionless → unchanged; far-field/energy quantities
restore `L0`. Sits upstream of equilibration and ①.

---

## Recommendation

Defer **①** into the eventual hierarchical-basis work (②); a 2× constant factor
does not justify a basis rewrite on its own. Neither lever touches the geometric
sliver blowup — that remains the own-mesher's job.

## ④ — IMPLEMENTED (characteristic-length non-dimensionalization)

Lever ④ is now in the solver. `Mesh::normalize_characteristic_length` divides
the node coordinates by `L0 = mean edge length` at `Simulation::new`; the
frequency state (`Excitation`) carries `κ = k0·L0` for the wave operator and
propagation constants while keeping the physical `ω` for the dispersive
material/circuit terms (length-coupled wave impedances use `ω̃ = κ·c0`). The
solver assembles and solves on O(1) coordinates; outputs are converted back at
the boundary (E ÷L0, H/curl ÷L0², coordinates ×L0, eigen-frequency ÷L0,
far-field done fully in physical units). `RAPIDFEM_NO_NORMALIZE=1` keeps physical
units (`l0 = 1`) as an escape hatch / bit-identity oracle.

**Empirically validated** (all on light fixtures, < 100k DOF):

- Normalization is **bit-identical** to the physical path on S-parameters
  (WR-90 single-solve and the 11-frequency iris sweep) — no regression.
- Field outputs (VTK E/coords) agree to **4e-9 relative** (machine precision;
  the residual is O(1)-vs-physical rounding, not a factor error).
- Cavity eigen-frequencies match exactly (8.559581 GHz … invariant).
- **The headline win:** a geometrically perfect WR-90 scaled to **20 pm**
  features (where the un-normalized solver collapses to total reflection,
  `|S11|=1`) now returns the bit-identical *correct* S-parameters. Scale
  invariance is total across the full range (20 Mm → 20 pm), not just the
  physically realistic part the equilibration already covered.

The motivation is forward-looking as much as present: a future float32 / GPU
solve path has far less headroom than float64, so O(1)-magnitude assembly (which
equilibration cannot fully substitute for at low precision) becomes load-bearing.
