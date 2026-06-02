# RapidFEM Time-Domain Backend — Compact Macromodel Plan

The natural successor to the ports work, and the strongest candidate for
a differentiated capability beyond the kernel-tuning treadmill: turn a
matrix-free TD multiport problem into a **compact MIMO state-space
model** and its broadband S-matrix. The same flow then lifts to **fast
parameter optimisation** through parametric MOR.

The pitch: the matrix-free TD operator sidesteps two walls at once that
the frequency-domain backend hits on large RFIC structures, the
direct-solve RAM wall (~500k DOF) and the explicit CFL limit (the
projection is matvec-based, not time-stepped). The reduced model is
then a handful of kilobytes and evaluates the S-matrix essentially for
free over a band. This is the established Krylov-macromodelling
discipline (PRIMA / PVL / SPRIM), adapted to our matrix-free DGTD.

## Goal

End-to-end from a meshed `ProblemTD` with N ports to:

- a passive MIMO reduced-order state-space model `(Â, B̂, Ĉ)` of order
  `r ~ tens to low hundreds`,
- its broadband N-port S-matrix `H(ω) = Ĉ(jωI - Â)⁻¹B̂` over an arbitrary
  frequency sweep,
- a parametric form for parameter optimisation, where evaluating a new
  design point is essentially free.

## Approach

The TD multiport system is the linear time-invariant

```
  dy/dt = A·y + B·u(t)         y is the n_dof state, u the N port inputs
  z(t)  = C·y                  z the N port modal projections
```

with `A` the matrix-free DG Maxwell operator, `B = [b_1, …, b_N]` the
port-injection vectors (one column per port, from `port_source`), and
`C` the modal-extraction operator (from `port_modal_projections`).

**Block-Krylov projection.** Build an orthonormal basis `V` (`n_dof x
r`) of a block-Krylov subspace seeded by `B`. The reduced system is

```
  Â = Vᵀ A V        B̂ = Vᵀ B        Ĉ = C V
```

The choice of expansion point governs accuracy: impulse moments (`s=∞`,
the `Aᵏ B` Krylov space) need only matvecs and naturally suit a
matrix-free operator; rational / multi-point Krylov at frequencies in
the band converges faster but needs `(sI - A)⁻¹` solves and is a later
phase.

**Frequency response.** `H(ω) = Ĉ(jωI - Â)⁻¹B̂` is one `r x r` complex
solve per frequency — microseconds at `r ~ 100`, a 1000-point band is
milliseconds.

**Passivity.** Plain Krylov projection does not preserve passivity. For
the lossless first-order curl-curl system, the operator is
skew-symmetric and naturally passive; a **structure-preserving**
projection (SPRIM-style: project E and H sub-vectors independently to
keep the block structure) carries this through. For the lossy case, a
passivity-enforcement perturbation may be needed.

**Parametric form.** For optimisation, build a parametrised reduced
basis (matrix interpolation across sample (Â, B̂, Ĉ) at design points,
or a global basis spanning the parameter range). A new design point
then costs only the free reduced-model evaluate, not a rebuild.

## Status — all phases planned

What is already in place (the foundation):

- The **matrix-free TD operator** (`MaxwellOperator::apply` /
  `apply_into`), CPU and GPU.
- **Ports** — rectangular waveguide TE_mn, lumped / TEM via the (0,0)
  sentinel; coax in progress as part of the ports extension. The port
  machinery exposes the columns of `B` (`port_source`) and the modal
  projections (`port_modal_projections`).
- **Single-start Krylov reduction** (`P4` of the GPU plan): a
  `ReducedModel` from one start vector, on CPU and GPU. Not block, not
  multi-point, no passivity guarantee, no MIMO frequency-response
  extraction. A useful starting point and a reusable Arnoldi helper.
- **`state_space` export** — assembles the operator as a sparse matrix
  for analysis and cross-checks, separate from the reduced flow.

Everything below is ahead.

## Phasing

### M1 — Block / multi-input macromodel build

| WP | Deliverable | Gate |
|----|-------------|------|
| 1.1 | Block-Arnoldi seeded by all port columns `B = [b_1,…,b_N]`; outputs `(Â, B̂, Ĉ)` of order `r`. Extend the existing `arnoldi` helper to block input. | A matched two-port straight guide: the reduced model reproduces the per-frequency `|S₁₁| ≈ 0`, `|S₂₁| ≈ 1` within a few percent. |

### M2 — MIMO frequency-response extraction

| WP | Deliverable | Gate |
|----|-------------|------|
| 2.1 | `H(ω) = Ĉ(jωI - Â)⁻¹B̂` over a band, returned as an N × N complex matrix per frequency. | An iris / discontinuity case agrees with FD `sweep` within discretisation error (~2-3 %). |
| 2.2 | Touchstone (`.s{N}p`) export of the S-matrix. | A circuit simulator (or `scikit-rf`) loads the file and reproduces the band. |

### M3 — Passivity preservation

| WP | Deliverable | Gate |
|----|-------------|------|
| 3.1 | SPRIM-style structure-preserving projection — E and H sub-vectors projected independently so the [E, H] block structure (and the lossless skew-symmetry) carries through. | A bounded-real / positive-real eigenvalue check on the reduced model passes for matched-line and resonant-cavity cases. |
| 3.2 | Passivity-enforcement perturbation for the lossy case (if needed by 3.1). | The model stays passive when small losses are introduced; accuracy degradation under perturbation is bounded. |

### M4 — Broadband accuracy refinement

Several matvec-only paths give broadband accuracy without ever needing
shift-invert. Try them in order of cost; only fall back to shift-invert
if none of them suffices.

| WP | Deliverable | Gate |
|----|-------------|------|
| 4.1 | **Just push `r` in impulse-Krylov.** On the GPU a single matvec is sub-ms and the O(r²·n) orthogonalisation at r ≈ 300-500 still fits the millisecond budget. For Maxwell systems with well-separated resonances, the impulse-Krylov projection captures the dominant in-band modes directly (Lanczos-style). | An octave-wide band hits M1 accuracy at `r ≤ 500`, build still under the GPU budget. |
| 4.2 | **Eigenvalue / Chebyshev-polynomial filtering.** Either extract the in-band eigenpairs of `A` (Lanczos) and use them as the reduced basis, or precondition the Krylov build with a Chebyshev polynomial filter focused on the band. Both are pure matvecs — no inversion. | Same accuracy gate as 4.1 at smaller `r`. |
| 4.3 | **Multi-point shift-invert, only if 4.1 / 4.2 are not enough.** Block-Arnoldi at a few expansion frequencies, with `(sI - A)⁻¹` applied via **matrix-free GMRES on `(sI - A)`** — still matvec-only, no assembled matrix needed. The HX / auxiliary-space ideas from `iterative-solver-research.md` are inspiration for preconditioning, but the codebase is independent — that research targets the assembled FD Nédélec matrix, not our matrix-free TD operator. | The shift-invert path reaches the same accuracy at substantially smaller `r` than 4.1, justifying its added complexity. |

### M5 — Python API and an RFIC-style example

| WP | Deliverable | Gate |
|----|-------------|------|
| 5.1 | `ProblemTD.macromodel(frequencies, ...)` Python verb. Returns a `MacroModel` object: the state-space `(Â, B̂, Ĉ)`, an `evaluate(ω)` to a complex N×N matrix, a `.touchstone(path)` writer. | Verb runs end-to-end from Python. |
| 5.2 | An RFIC-style example (spiral inductor or multi-port matching network) — geometry → meshed `ProblemTD` with lumped ports → macromodel → Touchstone, cross-validated against FD. | The example produces an S-matrix that agrees with FD within discretisation error. |
| 5.3 | Docs and a regression test. | CI green. |

### M6 — GPU acceleration of the build

| WP | Deliverable | Gate |
|----|-------------|------|
| 6.1 | Block-Arnoldi on GPU — extend `GpuOperator::arnoldi` to a block input, reusing the f64 CGS2 and the f32 apply matvec. | The build is sub-second on the GPU for a ~1M-DOF, few-port case (CPU build of the same model is seconds). |
| 6.2 | Memory care for the block basis — the same Krylov-chunk sub-stepping pattern (`KRYLOV_CHUNK` in `constants.rs`) applies if `r * N * n_dof` pressures GPU memory at large structures. | A ~10M-DOF case fits the GPU at a useful `r`. |

### M7 — Parametric MOR for fast optimisation

| WP | Deliverable | Gate |
|----|-------------|------|
| 7.1 | A parametrised reduced model — matrix interpolation between sample `(Â, B̂, Ĉ)` at design points, or a global reduced basis covering the parameter range. | A new parameter point evaluates at essentially the reduced-model cost (microseconds), not a rebuild. |
| 7.2 | An optimisation example (e.g. tune a matching network's dimensions for a target return loss). | The optimisation loop with pMOR is >10× faster than per-point rebuild. |

## Critical path

```
M1 ─▶ M2 ─▶ M3 ─▶ M5            (build → response → passive → Python)
              └▶ M4              (multi-point — accuracy refinement)
M5 ─▶ M6                         (GPU — acceleration once the method is proven)
M6 ─▶ M7                         (pMOR — optimisation, on the proven flow)
```

- **M1 + M2 unlock the basic flow.** Until a matched line reproduces
  S-parameters cleanly, the macromodel machinery is not trusted.
- **M3 is the danger zone** — like the port flux was for the ports
  plan. Get passivity right, or the model is unusable in a circuit
  simulator.
- **M5 is the deliverable milestone** — end-to-end RFIC example from
  Python, cross-validated against FD.
- **M4, M6, M7 are independent refinements** on the proven flow.

## Risks

- **Passivity is the core danger zone.** A non-passive reduced model
  destroys a circuit simulation. SPRIM-style structure-preserving
  projection is the principled answer; the implementation needs care.
- **Broadband accuracy with impulse-only Krylov can need large `r`** for
  resonant or high-Q structures. M4 (multi-point) is the answer; until
  it lands, the impulse-Krylov flow is bounded in band-to-resonance
  density.
- **Block-basis memory at large `r × N × n_dof`** is the next memory
  cliff. The sub-stepping pattern that worked for `expmv` lifts: cap the
  effective Krylov dimension per build phase and accumulate.
- **Broadband accuracy at very wide bands** may push impulse-Krylov to
  uncomfortable `r`. The matvec-only fallbacks (eigenvalue / Chebyshev
  filtering) usually suffice; shift-invert is a last resort and even
  then runs matrix-free via GMRES — no dependence on the assembled-FD
  iterative-solver project.

## Out of scope

- Nonlinear macromodels (this is LTI by construction).
- Time-varying or nonlinear materials.
- Multi-physics couplings beyond electromagnetics.
- New port types beyond what the ports extension provides.

## Branch and pattern

Continues on `master`, matching this session's pattern; incremental
commits per WP, each gated before the next.
