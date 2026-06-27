# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Clean-room derivation: robust quasi-TEM wave-port mode solve.

INDEPENDENT DERIVATION (nothing transcribed from any other solver). The 2-D
vector port eigenproblem itself is already assembled correctly in
crates/rapidfem-core/src/port_eigen.rs (mixed Nedelec-tangential Eₜ + Lagrange
longitudinal E_z, generalized EVP `A x = λ B x` with `λ = −β²`). What fails on
µm-scale RFIC cross-sections is the NUMERICS around it. Here we derive, from
primary scaling + spectral-transform theory, the four robustness fixes:

  (1) PORT-LOCAL non-dimensionalization — scale the 2-D solve by the PORT's own
      characteristic length ℓ (NOT the global mesh L0), so the matrix entries
      are O(1) and the conditioning is independent of how small the port is
      relative to the rest of the 3-D mesh. (rapidfem's own Lever ④, applied to
      the port instead of inheriting the global one.)
  (2) The shift-invert spectral map and WHY a single shift at εmax is wrong for
      inhomogeneous lines (it targets the curl-free spurious cluster, not the
      genuine quasi-TEM at εeff) — so the band is probed with a shift sweep.
  (3) SCALE-INVARIANT tolerances (relative, not absolute k0²-scaled).
  (4) Recovery n_eff = √(−λ̃)/κ, invariant to the choice of ℓ.

The IMPLEMENTED fix is (1) port-local non-dimensionalization (the scale-
invariance lever). The electrically-small κ≪1 RFIC regime is flagged OPEN — it
needs gradient-null-space deflation (divergence cleaning), out of scope here.

Conventions match the kernel: e^{+iωt}, e^{−iβz} propagation, λ = −β²,
κ = k0·ℓ (electrical size of the port length scale).
"""
from __future__ import annotations
import sympy as sp


def section(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# Symbols
l, k0, beta, ell, eps_r, eps_max = sp.symbols("l k0 beta ell epsilon_r epsilon_max", positive=True)
kappa = sp.symbols("kappa", positive=True)        # κ = k0·ℓ
lam, lamt = sp.symbols("lambda lambdat", real=True)  # λ (phys), λ̃ (port-normalized)


# ---------------------------------------------------------------------------
section("(1) Port-local non-dimensionalization of  A x = λ B x,  λ = −β²")
# ---------------------------------------------------------------------------
# The 2-D vector waveguide EVP has the schematic block form (port_eigen.rs):
#   A = S − k0²·M    (S = transverse curl-curl stiffness, M = ε-weighted mass)
#   B = mass-like,   eigenvalue λ = −β²   (propagating band λ < 0).
# Scale the in-plane coordinates by a length ℓ:  x = ℓ·x̃,  x̃ = O(1).
# The Nedelec basis is scale-invariant (φ = l·L_a·W with l~ℓ, W~1/ℓ ⇒ φ~O(1)),
# while  curl_t φ = (1/ℓ)·curl̃ φ  and  dA = ℓ²·dÃ.  Hence the entries scale as
S, M, Bmat = sp.symbols("S M B", positive=True)     # the O(1) dimensionless blocks (tilde)
# S_phys = ∫(curlφ)² dA ~ (1/ℓ²)·ℓ² = ℓ⁰ · S̃     (curl-curl is scale-free in 2-D)
# M_phys = ∫ε φ²   dA ~  1   ·ℓ² = ℓ² · M̃
# B_phys ~ ℓ²·B̃
S_phys = S
M_phys = ell**2 * M
B_phys = ell**2 * Bmat
A_phys = S_phys - k0**2 * M_phys
print("  A_phys = S − k0²·M_phys =", A_phys, "   (S̃ scale-free, M_phys=ℓ²·M̃)")
# A_phys x = λ B_phys x  ⇒  (S̃ − (k0ℓ)² M̃) x = (λ ℓ²) B̃ x
A_norm = sp.expand(A_phys.subs(M, M) )  # keep symbolic
# substitute κ = k0·ℓ and λ̃ = λ·ℓ²
A_tilde = S - kappa**2 * M
lhs = sp.Eq(sp.Symbol("A_tilde"), A_tilde)
print("  normalized:  (S̃ − κ²·M̃) x = λ̃ · B̃ x,   κ = k0·ℓ,  λ̃ = λ·ℓ²")
# λ = −β² ⇒ λ̃ = −β²ℓ² = −(βℓ)² = −β̃²,  β̃ = β·ℓ.
print("  λ = −β²  ⇒  λ̃ = −(β·ℓ)² = −β̃²")
# n_eff² = −λ/k0² = −λ̃/(k0ℓ)² = −λ̃/κ²   →  INVARIANT to ℓ.
neff2_from_phys = -lam / k0**2
neff2_from_norm = (-lamt / kappa**2)
# check consistency with λ̃ = λ ℓ², κ = k0 ℓ
chk = sp.simplify(neff2_from_norm.subs({lamt: lam*ell**2, kappa: k0*ell}) - neff2_from_phys)
print("  n_eff² = −λ/k0² = −λ̃/κ² :", "INVARIANT" if chk == 0 else f"MISMATCH {chk}")
assert chk == 0
print("  → choosing ℓ = PORT size makes x̃=O(1) and every block O(1); the global")
print("    L0 (≫ port for RFIC) does NOT, leaving κ tiny and S̃≫κ²M̃ ill-scaled.")


# ---------------------------------------------------------------------------
section("(2) Shift-invert spectral map — and why a single shift is NOT enough")
# ---------------------------------------------------------------------------
# Propagating modes have 0 < n_eff² ≤ εmax, i.e.  λ̃ = −κ²·n_eff² ∈ [−κ²εmax, 0).
# Shift-and-invert with shift σ maps λ̃ → μ = 1/(λ̃ − σ): the modes nearest σ are
# amplified most. Take σ = −1.1·εmax·κ² (just below the band) as a reference.
sigma = -sp.Rational(11, 10) * eps_max * kappa**2
neff2 = sp.symbols("neff2", positive=True)         # a mode's n_eff² ∈ (0, εmax]
lamt_mode = -kappa**2 * neff2
mu = sp.simplify(1 / (lamt_mode - sigma))
print("  σ = −1.1·εmax·κ² =", sigma)
print("  μ(n_eff²) = 1/(λ̃ − σ) =", mu)
dmu = sp.simplify(sp.diff(mu, neff2))
print("  dμ/d(n_eff²) =", dmu, " > 0  → μ monotone ↑ in n_eff² (largest n_eff² most amplified)")
assert dmu.subs({kappa: 1, eps_max: 1, neff2: sp.Rational(1, 2)}) > 0
mu_top = mu.subs(neff2, eps_max)         # mode at the TOP of the band (n_eff²=εmax)
mu_cutoff = mu.subs(neff2, 0)            # mode at cutoff (n_eff²→0)
ratio = sp.simplify(mu_top / mu_cutoff)
print(f"  μ(εmax)/μ(0) = {ratio}  → a shift at εmax amplifies the TOP of the band 11×")
assert sp.simplify(ratio - 11) == 0
print("""
  CAVEAT (the integrity correction — why σ=−1.1εmax·κ² alone is WRONG):
  For an INHOMOGENEOUS cross-section the genuine quasi-TEM mode sits at
  n_eff² = εeff < εmax, while the CURL-FREE SPURIOUS modes (gradient null-space
  of the curl-curl, E = ∇φ) sit at the material values n_eff² = εr, INCLUDING
  εmax. So the most-amplified mode at σ near εmax is the spurious εmax cluster,
  NOT the genuine quasi-TEM. A single shift there targets the wrong cluster; the
  genuine mode appears only when the Krylov reach spans down to εeff (moderate
  κ). Robustly: PROBE the band with several shifts (the implementation sweeps
  σ across fractions of εmax) and REJECT curl-free (k_t² ≈ 0) spurious modes.
  The electrically-small κ≪1 limit, where the genuine quasi-TEM is itself nearly
  curl-free and spectrally buried in the spurious cluster, is NOT solved by any
  single shift OR sweep — it needs gradient-null-space deflation (divergence
  cleaning), which is out of scope here.""")


# ---------------------------------------------------------------------------
section("(3) Recovery + TEM sanity")
# ---------------------------------------------------------------------------
# From the wanted eigenpair: λ̃ = σ + 1/μ ; β̃ = √(−λ̃) ; n_eff = √(−λ̃)/κ = β/k0.
lamt_rec = sigma + 1/mu                  # using μ above ⇒ should give λ̃_mode
lamt_rec = sp.simplify(lamt_rec)
print("  λ̃ = σ + 1/μ =", lamt_rec, " (== −κ²·n_eff² :",
      sp.simplify(lamt_rec - lamt_mode) == 0, ")")
neff_rec = sp.sqrt(-lamt_rec) / kappa
print("  n_eff = √(−λ̃)/κ =", sp.simplify(neff_rec), " (== √n_eff²)")
# Homogeneous fill εr: exact TEM mode β = k0√εr ⇒ n_eff = √εr, λ̃ = −κ²εr.
neff_tem = neff_rec.subs(neff2, eps_r)
print("  homogeneous εr  → n_eff =", sp.simplify(neff_tem), " (expect √εr):",
      sp.simplify(neff_tem - sp.sqrt(eps_r)) == 0)
assert sp.simplify(neff_tem - sp.sqrt(eps_r)) == 0


# ---------------------------------------------------------------------------
section("(4) Scale-invariant acceptance thresholds")
# ---------------------------------------------------------------------------
# The current code rejects modes with absolute, k0²-scaled floors
#   β² ≤ 1e-3·k0²   and   k_t² ≤ 1e-3·k0² .
# In the port-normalized variables these become PURE-NUMBER thresholds on n_eff²
# (β̃²/κ² = n_eff²) and on the transverse wavenumber fraction — independent of
# the port's physical size, so they no longer trip at µm scale:
neff2_floor = sp.Rational(1, 1000)       # accept propagating modes with n_eff² > 1e-3
print("  propagating test:   n_eff² = β̃²/κ² > 1e-3   (dimensionless, scale-free)")
print("  range test:         0 < n_eff² ≤ εmax + tol")
print("  Ritz/Arnoldi tol:   relative to ‖A_tilde‖ (O(1)), not absolute 1e-12")
print(f"  (was: β²≤1e-3·k0², k_t²≤1e-3·k0²  → both scale with k0², trip at µm scale)")


section("RESULT")
print("""Self-derived & validated:
  • port-local scaling (IMPLEMENTED): λ̃ = λ·ℓ², κ = k0·ℓ, n_eff = √(−λ̃)/κ,
    ℓ-invariant. Assemble the blocks in x̃ (ℓ = √port-area) so the conditioning
    and the acceptance thresholds are independent of the port's physical size.
    This is the fix that makes the solve scale-invariant.
  • recovery λ̃ = σ + 1/μ ; homogeneous fill → n_eff = √εr (exact).
  • shift-invert map μ = 1/(λ̃−σ) is monotone in n_eff² (each shift amplifies the
    modes nearest it) — but a SINGLE shift at εmax targets the curl-free spurious
    cluster, not the genuine quasi-TEM at εeff < εmax (inhomogeneous lines). The
    band is therefore probed with a shift sweep + curl-free spurious rejection.
  • OPEN: the electrically-small κ≪1 regime (real RFIC ports) is NOT solved here
    — the genuine quasi-TEM is nearly curl-free and buried in the spurious
    cluster, needing gradient-null-space deflation (divergence cleaning).""")
print("\nOK: wave-port scale-invariance derivation self-consistent "
      "(shift strategy + κ≪1 deflation flagged as open).")
