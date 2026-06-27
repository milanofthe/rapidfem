# Robust quasi-TEM wave-port mode solve

Clean-room derivation behind the numerical robustness of the 2-D vector
wave-port eigensolver (`crates/rapidfem-core/src/port_eigen.rs::solve_vector_modes`),
which previously returned **0 modes** on µm-scale RFIC cross-sections.
Run `python wave_port_modes.py`.

The eigenproblem (mixed Nédélec-tangential `Eₜ` + Lagrange longitudinal `E_z`,
`A x = λ B x`, `λ = −β²`) is already assembled correctly; the issue is the
numerics around it. Derived & validated here:

1. **Port-local non-dimensionalization** *(the implemented fix)* — scale the 2-D
   solve by the *port's* characteristic length ℓ (not the global mesh `L0`), so
   `κ = k0·ℓ`, the blocks are O(1), and the conditioning + acceptance thresholds
   are independent of how small the port is vs the 3-D mesh. `n_eff = √(−λ̃)/κ`
   is ℓ-invariant. This makes the solve **scale-invariant**.
2. **Shift-invert spectral map** `μ = 1/(λ̃−σ)` is monotone in `n_eff²` (each
   shift amplifies the modes nearest it). **A single shift at εmax is wrong for
   inhomogeneous lines**: the genuine quasi-TEM sits at `εeff < εmax`, while the
   curl-free *spurious* modes (gradient null-space, `E = ∇φ`) sit at the material
   `εr` values incl. εmax — so a shift at εmax targets the spurious cluster. The
   band is therefore probed with a shift sweep + curl-free spurious rejection.
3. **Scale-invariant thresholds** — pure `n_eff²` numbers, not absolute
   `k0²`-scaled floors that trip at µm scale.
4. **Unit-power normalization** `∫(E×H*)·n̂ dS = 1`.

**Open / out of scope:** the electrically-small `κ ≪ 1` regime (real RFIC ports)
is *not* solved by any single shift or sweep — there the genuine quasi-TEM is
itself nearly curl-free and spectrally buried in the spurious cluster, needing
**gradient-null-space deflation (divergence cleaning)**. RFIC passives are
meanwhile served by the (robust) lumped ports.

Textbook basis: shift-and-invert spectral transform (Saad, *Numerical Methods
for Large Eigenvalue Problems*) + the waveguide quasi-TEM mode (Pozar). No code
from any other solver was used.
