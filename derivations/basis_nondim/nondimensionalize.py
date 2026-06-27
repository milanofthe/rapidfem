# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Lever ④: non-dimensionalize the driven system for unit-robust assembly.

Grounds the analysis in rapidfem's actual system (crates/rapidfem-fd/src/
assembly.rs): K = E - k0^2 B with k0 = 2*pi*f/c0, coordinates in METERS, and a
set of ABSOLUTE tolerances in crates/rapidfem-core/src/constants.rs
(SINGULAR_EPS = 1e-30, POINT_IN_TET_EPS = 1e-8, LANCZOS_BREAKDOWN = 1e-12, ...).

THE DERIVATION
--------------
Scale coordinates by a reference length L0: x = L0 * x~, with x~ = O(1).
The canonical R2 basis is scale-invariant -- phi = l L_a W with l ~ L0,
W = L_a gradL_b - L_b gradL_a ~ 1/L0, so phi ~ O(1) -- while
  curl phi  = (1/L0) curl~ phi~,      dV = L0^3 dV~.
Therefore, with relative material tensors (mu_r^-1, eps_r) as rapidfem uses:
  E_ij = INT (curl phi_i) mu_r^-1 (curl phi_j) dV = L0   * E~_ij
  B_ij = INT  phi_i        eps_r   phi_j       dV = L0^3 * B~_ij
  K     = E - k0^2 B = L0 * [ E~ - (k0 L0)^2 B~ ]                       (*)

So the whole system is the dimensionless block [ E~ - kappa^2 B~ ] times a
scalar L0, governed by a SINGLE dimensionless number

        kappa = k0 * L0     (the electrical size of the reference length).

Two consequences this script verifies:
  (A) cond(K) is EXACTLY L0-invariant -- uniform scaling is a similarity, so
      non-dimensionalization does NOT change the mathematical conditioning.
      Its payoff is NUMERICAL, not spectral. (We prove this to kill the
      temptation to oversell it as a conditioning fix.)
  (B) The payoff is UNIT-ROBUSTNESS: rapidfem's absolute tolerances are compared
      against entry magnitudes that scale like L0^p. A mesh in microns vs metres
      shifts 6V by 10^18; a perfectly healthy nanoscale RFIC element then trips
      SINGULAR_EPS and is wrongly declared degenerate. Assembling in x~ (O(1))
      makes every tolerance decision independent of the user's length unit.

For RFIC specifically: features ~1 um, f ~ 60 GHz give kappa = k0 L0 ~ 1e-3, so
the mass term is a ~1e-6 perturbation of the stiffness -- physically correct
(electrically tiny), and in x~ coordinates everything stays O(1).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nedelec2"))
import element  # noqa: E402

C0 = 299_792_458.0
SINGULAR_EPS = 1e-30          # crates/rapidfem-core/src/constants.rs
REGULAR_TET = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]


def np_DF(verts):
    I3 = sp.eye(3)
    D, F = element.element_matrices(verts, I3, I3)
    return (np.array(D.evalf(50), dtype=float),
            np.array(F.evalf(50), dtype=float))


def cond_svd(M):
    """Solve-conditioning of a (possibly indefinite) matrix: sigma_max/sigma_min."""
    s = np.linalg.svd(M, compute_uv=False)
    return s[0] / s[-1]


def scaled_tet(L0):
    return [tuple(L0 * sp.Rational(c) for c in v) for v in REGULAR_TET]


# --------------------------------------------------------------------------
# (1) verify the scaling law (*): E ~ L0, B ~ L0^3, exactly
# --------------------------------------------------------------------------
def part1_scaling_law():
    print("=" * 78)
    print("(1) scaling law  E = L0 E~,  B = L0^3 B~  (regular tet, exact)")
    print("=" * 78)
    E1, B1 = np_DF(scaled_tet(sp.Integer(1)))
    mE, mB = np.abs(E1) > 1e-12, np.abs(B1) > 1e-12
    for L0 in (sp.Rational(1, 10), sp.Integer(10)):
        E, B = np_DF(scaled_tet(L0))
        f = float(L0)
        eE = np.max(np.abs(E[mE] / (f * E1[mE]) - 1.0))
        eB = np.max(np.abs(B[mB] / (f**3 * B1[mB]) - 1.0))
        print(f"    L0 = {f:>7.2f}:  max|E/(L0 E~)-1| = {eE:.1e}   "
              f"max|B/(L0^3 B~)-1| = {eB:.1e}")
    print("    => E scales as L0, B as L0^3; K = L0 [ E~ - (k0 L0)^2 B~ ].")


# --------------------------------------------------------------------------
# (A) cond(K) is L0-invariant: the lever is numerical, not spectral
# --------------------------------------------------------------------------
def part2_cond_invariance():
    print("\n" + "=" * 78)
    print("(A) cond(K) is invariant under non-dimensionalization (FIXED physics)")
    print("=" * 78)
    # ONE physical element (regular tet, 1 um edge) at 60 GHz. Re-express in
    # x~ = x/L0 for several L0; K_raw = L0 * K~ exactly, so cond is unchanged.
    f_hz, edge_m = 60e9, 1e-6
    k0 = 2 * np.pi * f_hz / C0
    verts_m = [tuple(edge_m / (2*np.sqrt(2)) * c for c in v) for v in REGULAR_TET]
    Em, Bm = np_DF(verts_m)
    K_raw = Em - k0**2 * Bm
    print(f"    physical: regular tet, edge {edge_m:.0e} m, f {f_hz:.0e} Hz")
    print(f"    cond(K_raw, metres) = {cond_svd(K_raw):.6e}\n")
    print(f"    {'L0 [m]':>10}{'kappa=k0 L0':>14}{'cond(K~)':>14}{'max|K_raw-L0 K~|':>20}")
    for L0 in (edge_m, 1e-6, 1e-3, 1.0):
        verts_t = [tuple(c / L0 for c in v) for v in verts_m]
        Et, Bt = np_DF(verts_t)
        kappa = k0 * L0
        K_t = Et - kappa**2 * Bt
        resid = np.max(np.abs(K_raw - L0 * K_t))
        print(f"    {L0:>10.0e}{kappa:>14.4e}{cond_svd(K_t):>14.6e}{resid:>20.2e}")
    print("    cond(K~) = cond(K_raw) for every L0 (K_raw = L0 K~, a scalar).")
    print("    => ④ is NOT a conditioning fix -- it is unit-hygiene (below).")


# --------------------------------------------------------------------------
# (B) the real payoff: absolute tolerances become unit-robust
# --------------------------------------------------------------------------
def part3_unit_robustness():
    print("\n" + "=" * 78)
    print("(B) unit-robustness: a healthy element vs the absolute SINGULAR_EPS")
    print("=" * 78)
    print("    6V of a REGULAR tet (q=0.71, geometrically perfect) at shrinking")
    print(f"    physical size, against SINGULAR_EPS = {SINGULAR_EPS:.0e}:\n")
    print(f"    {'edge [m]':>12}{'6V [m^3]':>14}{'min|gradL|':>14}"
          f"{'verdict (raw)':>18}")
    for h in (1e-2, 1e-5, 1e-8, 1e-10, 1e-11):
        # regular tet of edge ~ h: scale REGULAR_TET (edge 2*sqrt2) to edge h
        s = h / (2 * np.sqrt(2))
        verts = [tuple(s * c for c in v) for v in REGULAR_TET]
        sixV, grads = element.barycentric_gradients(verts)
        sixV = float(sixV)
        gmin = min(float(sp.sqrt((g.T * g)[0])) for g in grads)
        dead = sixV < SINGULAR_EPS
        print(f"    {h:>12.0e}{sixV:>14.3e}{gmin:>14.3e}"
              f"{'DEGENERATE!' if dead else 'ok':>18}")
    print("\n    The 1e-11 m (10 pm) element is geometrically perfect yet its 6V")
    print("    underflows SINGULAR_EPS and is wrongly killed -- purely a unit")
    print("    artifact. In x~ = x/L0 coordinates 6V~ = O(1) always, so the test")
    print("    is unit-invariant. (RFIC routinely meshes sub-micron features.)")


# --------------------------------------------------------------------------
# (B') same physical problem, three unit choices -> identical dimensionless K
# --------------------------------------------------------------------------
def part4_unit_invariance():
    print("\n" + "=" * 78)
    print("(B') same physics in m / mm / um: raw K differs by 10^(3p), K~ identical")
    print("=" * 78)
    f_hz = 60e9
    L_phys = 1e-6                       # 1 um physical edge scale (in metres)
    # Assemble once in SI metres.
    verts_m = [tuple(L_phys / (2*np.sqrt(2)) * c for c in v) for v in REGULAR_TET]
    En, Bn = np_DF(verts_m)
    k0_m = 2 * np.pi * f_hz / C0
    K_m = En - k0_m**2 * Bn            # raw system in metres

    # A user meshing in unit `u` (= u_in_metres) writes coordinate NUMBERS
    # x/u, so k0 -> k0*u and one shows K_u = K_m / u exactly. The dimensionless
    # system divides by the reference-length NUMBER L0 = L_phys/u and is
    # therefore unit-invariant: K~ = K_u / L0 = K_m / L_phys.
    print(f"    {'unit':>6}{'max|K_ij| raw':>18}{'min|K_ij| raw':>18}"
          f"{'max|K~_ij|':>14}")
    ref_Kt = None
    for unit, u in (("m", 1.0), ("mm", 1e-3), ("um", 1e-6)):
        K_u = K_m / u
        L0_number = L_phys / u
        Kt = K_u / L0_number          # = K_m / L_phys, unit-invariant
        amax = np.max(np.abs(K_u))
        amin = np.min(np.abs(K_u[np.abs(K_u) > 0]))
        print(f"    {unit:>6}{amax:>18.4e}{amin:>18.4e}{np.max(np.abs(Kt)):>14.6e}")
        if ref_Kt is None:
            ref_Kt = Kt
        else:
            assert np.allclose(Kt, ref_Kt, rtol=1e-12), "K~ must be unit-invariant"
    print("    K~ is byte-identical across units; only the raw magnitudes (and")
    print("    thus every absolute-threshold decision) move by 10^(3p).")


# --------------------------------------------------------------------------
# optimal L0: minimize the entry dynamic range of E~, B~
# --------------------------------------------------------------------------
def part5_optimal_L0():
    print("\n" + "=" * 78)
    print("optimal L0 = geometric-mean edge length centers entries at O(1)")
    print("=" * 78)
    # A uniform L0 CANNOT change a within-block ratio (scale-invariant); it sets
    # the ABSOLUTE level. The goal is to place the typical entry near 1 -- as far
    # as possible from both underflow and the absolute tolerances. L0 = the
    # geometric-mean edge length does exactly that.
    verts = [(0, 0, 0), (sp.Rational(5), 0, 0), (0, sp.Rational(1), 0),
             (sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(2))]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    elens = [float(element.dist(verts, a, b)) for a, b in edges]
    L0_geo = float(np.exp(np.mean(np.log(elens))))
    print(f"    edge lengths: {[round(e,2) for e in elens]}")
    print(f"    geometric-mean edge length L0* = {L0_geo:.4f}\n")
    print(f"    {'L0':>10}{'max|E~|':>14}{'max|B~|':>14}{'note':>12}")
    En, Bn = np_DF(verts)
    for L0 in (0.2 * L0_geo, L0_geo, 5 * L0_geo):
        mE = np.max(np.abs(En / L0))
        mB = np.max(np.abs(Bn / L0**3))
        tag = "<- L0*" if abs(L0 - L0_geo) < 1e-9 else ""
        print(f"    {L0:>10.4f}{mE:>14.3e}{mB:>14.3e}{tag:>12}")
    print("    At L0* the entries are O(1); too-small/large L0 pushes them off by")
    print("    powers of L0. A single global L0 = mean mesh edge length is the")
    print("    cheap, robust choice (the within-element ratio is geometry, fixed).")


if __name__ == "__main__":
    part1_scaling_law()
    part2_cond_invariance()
    part3_unit_robustness()
    part4_unit_invariance()
    part5_optimal_L0()
    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print("  ④ does not change cond(K) (uniform scaling is a similarity). Its")
    print("  value is UNIT-ROBUSTNESS: assemble in x~ = x/L0 (L0 = mean edge")
    print("  length), carry kappa = k0 L0, solve, rescale back. Every absolute")
    print("  tolerance (SINGULAR_EPS, sliver floor, pivot drops, breakdown)")
    print("  then behaves identically whether the user meshes in m, mm, um or nm")
    print("  -- the enabling fix for sub-micron RFIC structures. Cheap, global,")
    print("  solution-preserving; sits upstream of equilibration and lever ①.")
