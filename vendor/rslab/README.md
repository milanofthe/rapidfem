# RSLAB

Rust Sparse Linear Algebra Backend. A sparse direct solver for real and complex
matrices with **three paths matched to their operator classes**: symmetric LDLᵀ
(Bunch-Kaufman), unsymmetric LU, and a KLU path for circuit-shaped matrices —
with the factor usable as a preconditioner. The solver core is pure Rust with no
BLAS, LAPACK, or MKL dependency.

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

RSLAB factors `Pᵀ A P = L D Lᵀ` (complex-symmetric, PARDISO `mtype 6`),
`Pᵀ A P = L U` (unsymmetric, `mtype 13`), or a BTF block factorization
(circuit-shaped, KLU-style), then solves against one or many right-hand sides.
It is a fork of [feral](https://github.com/jkitchin/feral); see [NOTICE](NOTICE).
The accompanying technical report ([`docs/report/rslab.pdf`](docs/report/rslab.pdf))
derives the algorithms and carries the full evaluation; the numbers below are its
headline results.

## Features

- Pure-Rust solver core. No native dependencies. Optional bench/tooling features
  may load external libraries; the library does not.
- Generic over scalar type: `f64`, `f32`, `Complex<f64>`, `Complex<f32>`. A test
  factors and solves all four through both paths.
- Symmetric LDLᵀ with Bunch-Kaufman 1x1/2x2 pivoting (stores only `L`), and
  threshold-pivoted LU (exposed, tunable tolerance `u`) for unsymmetric matrices.
- **KLU path** for circuit-shaped matrices (`KluSymbolic::analyze → factor →
  KluSolver`): BTF (maximum transversal + Tarjan SCC, detects structural
  singularity a-priori) + per-block AMD + left-looking Gilbert-Peierls LU with
  threshold pivoting and row scaling. Strictly sequential and bit-deterministic;
  numeric-only `refactor` (frozen pattern + pivots) for frequency sweeps and
  Newton steps, plus `solve_transpose` (`Aᵀx = b` on the same factors) for
  adjoint / sensitivity solves. On MNA-like matrices: 2-19x faster factor with
  1.7-5.7x less fill than the multifrontal LU (widening with size), and a
  20-point same-pattern sweep 6-19x faster end to end
  (`cargo bench --bench klu_circuit`).
- Three factorization schedules: supernodal left-looking (default, frees each dense
  panel after its last consumer), multifrontal, and right-looking.
- Fill-reducing orderings: AMD, AMF, nested dissection (METIS/Scotch/KaHIP), and
  RCM (band/profile), selectable or raced per matrix.
- Tunable equilibration (one-pass ∞-norm, iterative Ruiz, MC64 matching, off) and
  factor emit/memory mode, all through one flat `SolverSettings` interface.
- Learned auto-tuner, **one model per path** (symmetric LDLᵀ / unsymmetric LU):
  a small MLP selects the solver configuration (ordering incl. `MetisND`, method,
  amalgamation, threshold-pivot `u` on LU, equilibration, memory mode, kernel
  gates) per matrix from its structural features, guarded by a deterministic
  a-priori memory backstop so it never uses more memory than the default;
  out-of-distribution it falls back to a deterministic exact-fill ordering race.
  Trained on a complete-distribution corpus including generated curl-curl Maxwell
  (complex indefinite), Stokes/KKT saddle-point, and convection-diffusion (the
  unsymmetric LU class, swept over the grid-Péclet range) systems.
- **Runtime tuner profile** (no recompile): the two models plus hardware-calibrated
  guard thresholds ship as a `tuner_profile.json` config artifact. Point
  `RSLAB_TUNER_PROFILE` at one (or call `apply_profile`) to specialize the tuner to
  a machine or problem class. Produced by the **meta-tuner** `cargo xtask tune`
  (sweep → train → hardware-calibrate → assemble → held-out validate), which only
  writes a profile that passes a **ship-gate** (must not regress the shipped default
  on a held-out generator corpus). Calibration sets the deviate guard to the
  machine's own timing noise floor (`z·CV`), so the tuner never chases a predicted
  gain smaller than the measurement variance.
- The numeric factor is bit-identical across thread counts; the parallel multi-RHS
  solve (8-19x faster than per-column) is bit-identical to the serial path.
- 32-bit index compression (`CompressedLdltFactors`, when `n < 2^31`): half the
  index footprint at no accuracy cost.
- Static pivot reuse for fixed-pattern value sequences (frequency sweeps, time
  stepping): skip the pivot search across refactorizations.
- Preconditioner mode: static pivoting (never-fail), optional incomplete drop and
  block-low-rank compression.
- Iterative solvers: flexible restarted GMRES (single + block/multi-RHS), COCG,
  COCR, with warm start (`x0`) and GCRO-DR Krylov subspace recycling (a `Recycle`
  handle carried across a sequence of related solves) for solver-in-the-loop work.
- A-priori peak-memory and runtime estimates computed from the symbolic structure
  before any numeric work; scoped per-solve thread pools; per-call diagnostics; an
  optional hardware-aware budget planner.

## Benchmarks

All cross-solver figures come from the `bench_suite` engine over a
complete-distribution corpus — structured-grid generators (curl-curl Maxwell,
shifted Helmholtz, Stokes/KKT saddle-point, convection-diffusion over the
grid-Péclet range, BEM/MoM near-field kernels; `src/matgen/fem.rs`) plus the
complex SuiteSparse matrices, 8k-125k DOFs, all `Complex<f64>` — measured in one
run on a quiet 12-core machine, so the cross-solver ratios carry no run-to-run
drift. RSLAB runs its auto-tuned default; each path is compared **on its own
class** against its own MKL PARDISO mtype and
[faer](https://github.com/sarah-quinones/faer-rs).

Reproduce: `RLA_BENCH_FAMILY=sym|unsym cargo bench --bench bench_suite
--features matgen`, then `benches/head_to_head.py`; the KLU comparison is
`cargo bench --bench klu_circuit`.

### Per-path scaling: RSLAB vs faer vs MKL PARDISO

Factor time and peak memory vs nonzeros, log-log, one power-law fit per solver.
Each plot carries two RSLAB curves — the **untuned default** (gray) and the
**auto-tuned** solver as shipped (blue) — so the gap the learned tuner closes
toward PARDISO is visible; it widens with problem size (a mispicked ordering
costs most on the big matrices) and never comes at a memory cost.

**LDLᵀ path (symmetric, PARDISO mtype 6)** — factor time (left) and peak memory (right):

![LDLt factor time (left) and peak memory (right)](benches/bench_out/h2h_ldlt.png)

**LU path (unsymmetric, PARDISO mtype 13)** — factor time (left) and peak memory (right):

![LU factor time (left) and peak memory (right)](benches/bench_out/h2h_lu.png)

Head-to-head geomean ratios (~100 matrices per path, 5k-1M nonzeros, over the
matrices both solvers factor to `< 0.1` residual):

| RSLAB (auto-tuned) vs | LDLᵀ (sym) | LU (unsym) |
|-----------------------|:----------:|:----------:|
| **MKL PARDISO** — factor time | 7.0x slower | **4.0x slower** |
| **MKL PARDISO** — peak memory | 2.2x more | 2.4x more |
| **faer LU** — factor time | **14.5x faster** | **6.7x faster** |
| **faer LU** — peak memory | **2.3x less** | 1.1x less |
| **untuned default** — factor time | **1.94x faster** | **1.78x faster** |
| **untuned default** — peak memory | **0.68x** (less) | **0.72x** (less) |

RSLAB sits between the two: faster and lighter than the pure-Rust faer,
moderately behind the hand-optimized MKL PARDISO, with the unsymmetric path the
closer of the two. faer has no symmetric path (it factors symmetric matrices as
LU too), so its LDLᵀ gap is structurally largest; it also OOMs on the largest
matrices, so its head-to-head is a conservative floor. On time the LU path
scales slightly flatter than PARDISO (`α≈1.17` vs `1.22`).

### Accuracy (SuiteSparse)

![SuiteSparse residual](benches/bench_out/corpus_residual.png)

Relative residual `‖Ax-b‖/‖b‖` as the accuracy check across the corpus.

- Where RSLAB factors, it is accurate: 24/31 matrices below `1e-8` residual, matching
  PARDISO and ahead of faer, which returns a degraded or garbage solution on several
  (pdb1HYS, bcsstk18, msc10848, wang3).
- Exact-mode limit: RSLAB's exact LDLᵀ (pivoting bounded to each supernode) cannot
  factor some indefinite saddle-point / KKT matrices (stokes64, bratu3d, cont-201) that
  PARDISO factors directly; it declines them rather than returning a degraded solution.
- Preconditioner mode covers most of that gap: a never-fail static-pivot factor used as
  a GMRES preconditioner reaches 28/33 below `1e-8` (matching PARDISO) and rescues the
  exact-mode failures bratu3d and cont-201; it also refines RSLAB's one inaccurate exact
  solve (qc2534, `3.6e-4` to `1.8e-13`). The hardest saddle-point/CFD cases (stokes64,
  ex11) stay out of reach. RSLAB targets the complex-symmetric EM/FEM regime, not
  general indefinite KKT.

The determinism and equivalence properties were validated over 180 SuiteSparse
matrices: right-looking vs multifrontal, both emit modes, parallel vs serial
front subtraction, and the 32-bit compressed factor are all bit-identical.

### KLU path on circuit-shaped matrices

KLU vs the multifrontal LU (defaults) on MNA-like matrices — ~4-5 nnz/column,
unsymmetric, column-diagonally dominant, cascaded stages giving a genuinely
reducible BTF structure (`cargo bench --bench klu_circuit`, Apple M3):

| n | nnz | KLU factor | KLU refactor | KLU fill | BTF blocks | MF-LU factor | MF-LU fill | sweep ratio |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2k | 15k | 6.4 ms | 1.8 ms | 79k | 8 | 12.6 ms | 132k | 6.2x |
| 10k | 73k | 16.7 ms | 5.0 ms | 439k | 16 | 53.2 ms | 1.03M | 10.3x |
| 50k | 366k | 98.1 ms | 29.4 ms | 2.32M | 32 | 305 ms | 9.21M | 11.3x |
| 200k | 1.47M | 361 ms | 117 ms | 9.17M | 64 | 1.84 s | 52.2M | 19.1x |

The KLU factor is 2-19x faster with 1.7-5.7x less fill, the gap widening with
size as the multifrontal fronts grow; the numeric-only refactor runs ~3x faster
still, so a 20-point sweep (refactor+solve vs factor+solve — the "sweep ratio")
is 6-19x faster end to end. Both solvers reach machine-precision residuals
(~1e-15) on every size.

### The learned auto-tuner

`LdltSolver::factor` / `LuSolver::factor_auto` select the whole `SolverSettings`
vector (ordering incl. `MetisND`, method, amalgamation, threshold-pivot `u` on
LU, equilibration, memory mode, kernel gates) from the matrix's structural
fingerprint — one MLP per path, trained offline on the corpus knob sweep and
embedded for pure-Rust inference. Its picks are constrained by a deterministic
guard stack — an out-of-distribution fallback to the exact-fill ordering race, a
re-analysis check that the pick's exact fill/flops/memory floor stay within
`1.02x`/`1.05x`/`1.0x` of the default's, and a minimum-improvement threshold —
so **peak memory is guaranteed never to exceed the untuned default** while time
is optimized in aggregate:

| path | factor speedup | peak-memory ratio |
|---|:---:|:---:|
| **LDLᵀ** (167 matrices) | **1.33x** | 0.83x |
| **LU** (97 matrices) | **1.92x** | 0.72x |

(vs the untuned default, geomean, each path on its own class.) The runtime tuner
profile (`tuner_profile.json`, `RSLAB_TUNER_PROFILE` / `apply_profile`) ships the
two models plus hardware-calibrated guard thresholds; the meta-tuner
`cargo xtask tune` reproduces it (sweep → train → calibrate → assemble →
held-out ship-gate).

### A-priori predictors

![Memory estimate vs measured](benches/bench_out/memory_breakdown.png)

RSLAB predicts the factor-memory peak from the symbolic analysis alone, before
any numeric work, with a separate model per path: the left-looking panel-freeing
simulation (live panels + factor + input/scratch) and the multifrontal
level-parallel model (fronts plus live contribution blocks). Over the corpus both
bounds hold at an estimate/measured ratio of **~1.3 in geomean and never
under-predict**, so either is safe to compare against RAM for fail-fast
scheduling; the panel-freeing floor is the tighter quantity the tuner's memory
veto uses. The KLU path carries the same contract (a pattern-only
Gilbert-Peierls pass gives its fill and flops exactly under diagonal pivoting).

The thread-aware runtime estimate combines the calibrated machine throughput
with an Amdahl critical-path floor from the assembly tree (a learned additive
residual on the speedup curve cuts the held-out error ~26%). The
`Threads::Auto` predictor lands within ~10% of the per-matrix-optimal worker
count (geomean) against ~50% for a fixed budget of 2, which is why the default
caps at 4 workers — the pareto-optimal throughput-per-core point.

### Iterative layer

The Krylov results, measured with their concepts: block-CGS2 lifts multi-RHS
strong scaling to **~2.2x at 12 cores** where per-RHS MGS is flat-to-negative
(preconditioned convection-diffusion, n=40000, complex), with per-RHS cost
near-flat in the block width; within-cycle deflation compacts the batched
operator applies to **0.66x** the full-width bound at `s=16`; GCRO-DR recycling
cuts the cross-solve iteration total **6.4x** on a stagnating 8-solve sequence
(2.9x on the first solve alone) for a ~1.5x wall-clock win; and the
incomplete-factor sweet spot at `drop_tol=1e-2` **halves the factor memory** at
a total wall time within a few percent of the exact direct solve. FGMRES's
flexible-basis update saves exactly one preconditioner solve per restart cycle;
the parallel `solve_many` behind the block preconditioner applies is 8-19x
faster than per-column. All of it stays bit-identical across thread counts.

## Install

```toml
[dependencies]
rslab = "0.17"
```

### Python (NumPy / SciPy)

```bash
pip install rslab
```

```python
import numpy as np, scipy.sparse as sp, rslab
x = rslab.spsolve(A, b)              # one-shot (auto symmetric/unsymmetric)
f = rslab.ldlt(A); x = f.solve(b)    # factor once, solve many; also rslab.lu(A)

k = rslab.klu(A_circuit)             # circuit-shaped: BTF + Gilbert-Peierls
A_circuit.data *= 1.5                # sweep: same pattern, new values
k.refactor(A_circuit.data)           # numeric-only refactor, then solve again
xt = k.solve_transpose(b)            # A^T x = b on the same factors (adjoint)
```

A thin wrapper over the Rust core; the matrix dtype selects the field
(`float64`/`float32` real, `complex128`/`complex64` complex). All factor knobs
are keyword arguments (`threads`, `preconditioner`, `drop_tol`, `method`,
`memory` on `ldlt`/`lu`; `pivot_tol`, `row_scaling`, `btf` on `klu`). See
[`python/README.md`](python/README.md).

## Usage

### Symmetric direct solve (LDLᵀ)

```rust
use rslab::prelude::*;

// Real symmetric, lower triangle (i >= j).
let a = CscMatrix::<f64>::from_triplets(3, &[0, 1, 2, 1], &[0, 1, 2, 0],
                                        &[2.0, 2.0, 2.0, -1.0])?;
let sym = LdltSymbolic::analyze(&a)?;            // phase 1: analyze pattern once
let f   = sym.factor(&a, &FactorOptions::default())?;  // phases 2-3: factor
let x   = f.solve(&[1.0, 2.0, 3.0])?;            // solve A x = b
# Ok::<(), rslab::RslabError>(())
```

### Unsymmetric direct solve (LU)

```rust
use rslab::prelude::*;
use num_complex::Complex;

let c = |re, im| Complex::new(re, im);
let a = GeneralCsc::from_triplets(2, &[0, 1, 0, 1], &[0, 1, 1, 0],
                                  &[c(2., 0.), c(2., 0.), c(1., 0.), c(-1., 0.)])?;
let f = LuSymbolic::analyze(&a)?.factor(&a, &FactorOptions::default())?;
let x = f.solve(&[c(1., 0.), c(0., 1.)])?;
# Ok::<(), rslab::RslabError>(())
```

### Circuit-shaped direct solve (KLU)

```rust
use rslab::prelude::*;

# fn demo(a: &GeneralCsc<f64>, a2: &GeneralCsc<f64>, b: &[f64]) -> Result<(), rslab::RslabError> {
// BTF + per-block AMD + Gilbert-Peierls LU; strictly sequential, bit-deterministic.
let sym   = KluSymbolic::analyze(a)?;                  // pattern once (BTF + AMD + symbolic)
let est   = sym.estimate_memory::<f64>();              // a-priori, before numeric work
let mut f = sym.factor(a, &KluSettings::default())?;
let x  = f.solve(b)?;
let xt = f.solve_transpose(b)?;                        // A^T x = b (adjoint/sensitivity)
f.refactor(a2)?;                                       // same pattern, new values: no pivot search
let x2 = f.solve(b)?;
# let _ = (est, x, xt, x2); Ok(()) }
```

### Preconditioned iteration

```rust
use rslab::prelude::*;
# use num_complex::Complex;
# let c = |re, im| Complex::new(re, im);
# let a = CscMatrix::<Complex<f64>>::from_triplets(3, &[0,1,2,1], &[0,1,2,0],
#     &[c(4.,1.), c(4.,1.), c(4.,1.), c(-1.,0.2)])?;
// Static pivoting + incomplete drop give a never-fail preconditioner.
let opts = FactorOptions::preconditioner(1e-8).with_drop_tol(1e-2);
let m = LdltSolver::factor_with(&a, &opts)?;
let b = vec![c(1.0, 0.0); 3];
let res = cocg(&a, &b, &m, 1e-10, 100)?;
assert!(res.converged);
# Ok::<(), rslab::RslabError>(())
```

## API reference

### Phased workflow

Analyze-once, factor-many (PARDISO phases):

| Phase | Symmetric | Unsymmetric | Circuit-shaped |
|-------|-----------|-------------|----------------|
| 1: analyze pattern | `LdltSymbolic::analyze(&a)` | `LuSymbolic::analyze(&a)` | `KluSymbolic::analyze(&a)` |
| 2-3: factor values | `sym.factor(&a, &opts)` -> `LdltSolver<T>` | `sym.factor(&a, &opts)` -> `LuSolver<T>` | `sym.factor(&a, &settings)` -> `KluSolver<T>` |
| solve | `f.solve(&b)` / `f.solve_many(&b, nrhs)` | `f.solve(&b)` / `f.solve_many(&b, nrhs)` | `f.solve(&b)` / `f.solve_many(&b, nrhs)` / `f.solve_transpose(&b)` |
| re-factor same pattern | `sym.factor(&a2, …)` | `sym.factor(&a2, …)` | `f.refactor(&a2)` (numeric-only, frozen pivots) |

One-shot: `LdltSolver::factor(&a)` / `LuSolver::factor(&a, &opts)` /
`KluSolver::factor(&a, &settings)`.

### FactorOptions

| Method | Effect |
|--------|--------|
| `preconditioner(floor)` / `exact()` | static-pivot preconditioner vs fail on singular pivot |
| `with_drop_tol(τ)` | drop fill below relative `τ` (incomplete factor) |
| `with_blr(BlrMode::…)` | block-low-rank compression of large fronts |
| `with_method(FactorMethod::…)` | `LeftLooking` (default) or `Multifrontal` |
| `with_threads(n)` | scoped pool of exactly `n` workers (`0` = all cores) |
| `with_thread_policy(Threads::…)` | `Auto{max}` (predict per matrix, capped; **default `max:4`**), `Fixed(n)`, or `Ambient` (use the current pool — no new spawn) |
| `with_memory(MemoryMode::…)` | transient-memory strategy |

The factor is bit-identical regardless of `threads`; the thread count affects time
and transient working set, not the result. The default caps at **4 workers** — the
pareto-optimal throughput-per-core point (the efficiency knee is ~4–6 threads) and
the safe default for concurrent / embedded use.

### Solver handles

```rust
# use rslab::prelude::*;
# fn demo(f: &LdltSolver<f64>, b: &[f64]) -> Result<(), rslab::RslabError> {
let x  = f.solve(b)?;                 // single RHS
let xs = f.solve_many(b, 4)?;         // 4 RHS at once (row-major n x nrhs)
let nnz = f.factor_nnz();             // fill (nnz of L, or L+U)
let d = f.diagnostics();              // per-call factor diagnostics
# Ok(()) }
```

### Diagnostics

`solver.diagnostics()` returns per-call, concurrency-safe data (no global state):
measured factor time, fill, thread count, and the a-priori `MemoryEstimate`.

### A-priori estimate

`sym.estimate_memory::<T>()` is a deterministic function of the analyzed structure,
callable before any numeric work:

```rust
use rslab::prelude::*;
use num_complex::Complex;
# fn demo(a: &CscMatrix<Complex<f64>>) -> Result<(), rslab::RslabError> {
let sym = LdltSymbolic::analyze(a)?;
let est = sym.estimate_memory::<Complex<f64>>();
let runtime_ms = est.est_runtime_ms(2.0, 4.0);   // gflops, parallel speedup
if !est.fits_in(8 << 30) { /* over 8 GiB */ }
# Ok(()) }
```

### Iterative solvers

`gmres`, `gmres_block`, `cocg`, `cocr` over any `LinearOperator` + `Preconditioner`.
A factor implements `Preconditioner`. A `Complex<f32>` factor can precondition an
`f64` GMRES via `LowPrecisionPreconditioner`.

`gmres_block` drives `s` right-hand sides in lockstep and orthogonalizes the whole
panel with **block-CGS2** — a parallel, panel-wide sweep instead of per-RHS
Gram-Schmidt — so the multi-RHS solve now scales across threads (~2.5x at 12 cores
on a deep-Krylov solve, where the old per-RHS path was flat) while staying
bit-identical across thread counts.

**Solver-in-the-loop thread capping.** The block orthogonalization runs on the
ambient rayon pool, so cap the whole solve with one pool: factor once (its own
bounded, `Auto{max:4}` pool), then run the RHS loop inside `with_threads(4, …)`:

```rust
# use rslab::{factor_general_lu, gmres_block, with_threads, SolverSettings, RslabError};
# use rslab::sparse::general::GeneralCsc;
# fn demo(a: &GeneralCsc<f64>, batches: &[Vec<f64>], s: usize) -> Result<(), RslabError> {
let lu = factor_general_lu(a, &SolverSettings::default())?;   // Auto{max:4}
with_threads(4, || {
    for rhs in batches { let _ = gmres_block(a, rhs, s, &lu, 1e-8, 400, 80)?; }
    Ok::<_, RslabError>(())
})?;
# Ok(()) }
```

Both phases stay on 4 cores with no per-call thread spawn. To also re-factor on the
shared pool (e.g. every Newton step), pass `Threads::Ambient` in the settings.

### Tuning (feature `tuning`)

```rust
# #[cfg(feature = "tuning")]
# fn demo(sym: &rslab::LuSymbolic) {
use rslab::tuning::{HardwareInfo, Calibration, Budget, plan};
let hw    = HardwareInfo::probe();              // cores + RAM
let calib = Calibration::load_or_measure(&hw);  // measured throughput, cached to disk
let est   = sym.estimate_memory::<f64>();
let budget = Budget { max_mem_bytes: Some(4 << 30), allow_mixed_precision: true,
                      allow_drop_tol: Some(1e-3), ..Default::default() };
let p = plan(&est, &budget, &hw, &calib);
// p.opts, p.use_mixed_precision, p.est_peak_bytes, p.est_runtime_ms, p.fits, p.note
# }
```

`plan` is a pure function of `(estimate, budget, hw, calibration)`. The thread
count it picks is cost-model-driven: the fewest cores that reach the near-minimum
predicted time, using an Amdahl critical-path floor from the assembly tree, so it
stops adding workers once the serial critical path (not total work) dominates. A
small learned residual (`benches/fit_residual.py`, `amdahl_frac`-driven, ~26%
held-out error reduction) refines the analytical speedup curve; it is additive on
the calibrated base and floored at the critical path, so it never extrapolates a
true chain into an impossible speedup.

### Meta-tuner (`cargo xtask`)

The offline pipeline that produces a `tuner_profile.json` (feature `tuning`):

```
cargo xtask calibrate                       # hardware microbench summary
cargo xtask tune   <workdir>                # sweep -> train -> profile -> ship-gate
cargo xtask profile <models_dir> <out> [class]   # assemble + ship-gate only
cargo xtask validate <profile.json>         # held-out geomean speedup vs default
```

`tune` runs the corpus sweep, trains the two per-path models, measures this
machine's calibration, assembles a candidate profile, and validates it on a
held-out generator corpus (curl-curl + saddle-point). The **ship-gate** writes the
profile only if it does not regress the shipped default. Load the result at runtime
with `RSLAB_TUNER_PROFILE=<path>` — no recompile.

### Test-matrix generators (feature `matgen`)

```rust
# #[cfg(feature = "matgen")]
# fn demo() {
use rslab::matgen::{self, stencil, bem};
let a = stencil::laplacian::<f64>(&[64, 64, 64], &stencil::StencilOpts::default());
let k = bem::kernel(8000, &bem::BemOpts::default());
for spec in matgen::catalog() { let _ = spec.name; }
# }
```

## Architecture

- Ordering: nested dissection (METIS/Scotch) with an AMD/AMF fallback selected by a
  size/structure heuristic.
- Left-looking supernodal (default): each panel pulls BLAS-3 updates from its
  factored descendants, then a blocked in-place panel factorization (Bunch-Kaufman
  for LDLᵀ, threshold partial pivoting for LU). Panels are compacted and freed once
  their last consumer is done.
- Multifrontal (opt-in): assembly tree of dense fronts.
- Parallelism: rayon over the assembly tree plus a SIMD (`gemm`) Schur update, in a
  scoped pool. Thread scaling saturates early because work concentrates in a few
  large top-of-tree supernodes.

## Determinism and scalar genericity

The `analyze -> factor -> solve` pipeline is generic over `Scalar`
(`f64`/`f32`/`Complex<f64>`/`Complex<f32>`); the estimator scales with
`size_of::<T>()`. The factor's `L`/`U`/`D` are bit-identical for any thread count,
and the estimates are pure functions of the symbolic structure.

## Cargo features

| Feature | Adds |
|---------|------|
| (default) | solver core, pure Rust |
| `matgen` | test-matrix generators + catalog |
| `matgen-download` | SuiteSparse / Matrix Market fetcher (pure-Rust HTTP/gzip/tar) |
| `tuning` | hardware probe + calibration cache + budget planner (pulls `sysinfo`) |

## License

MIT, Copyright (c) 2026 Milan Rother. RSLAB is a fork of feral
(https://github.com/jkitchin/feral), Copyright (c) 2026 John Kitchin, also MIT.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).
