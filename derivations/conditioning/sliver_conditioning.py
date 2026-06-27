# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Local conditioning of the canonical R2 element under sliver degeneration,
and the solution-preserving remedies (diagonal equilibration + iterative
refinement). Grounds the implementation in `crates/rapidfem-fd`.

Reuses the canonical R2 element from derivations/nedelec2/element.py, so the
conditioning numbers are for the *exact* element rapidfem assembles.

Three parts:
  (1) Sweep a tetrahedron from regular to sliver; show how 6V, cond(D),
      cond(F) and cond(A = D + F) blow up — this fixes the floor threshold.
  (2) Diagonal (Jacobi) equilibration A -> S A S and a Ruiz sweep: prove it
      preserves the solution and quantify the conditioning gain.
  (3) Iterative refinement: derive and demonstrate the error contraction that
      recovers the digits a poorly conditioned factorization loses.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nedelec2"))
import element  # noqa: E402  (canonical R2 element matrices)


# --------------------------------------------------------------------------
# geometry: a tet that flattens into a sliver as h -> 0
# --------------------------------------------------------------------------
def sliver_tet(h):
    """Right-triangle base in z=0, apex lifted by height h. As h->0 all four
    nodes become coplanar (a sliver) while edge lengths stay O(1)."""
    return [(0, 0, 0), (1, 0, 0), (0, 1, 0),
            (sp.Rational(1, 3), sp.Rational(1, 3), h)]


def six_volume(verts):
    M = sp.Matrix([[sp.Integer(1), v[0], v[1], v[2]] for v in verts])
    return abs(M.det())


def mean_edge(verts):
    import itertools
    ls = []
    for a, b in itertools.combinations(range(4), 2):
        d = sp.Matrix(verts[a]) - sp.Matrix(verts[b])
        ls.append(sp.sqrt((d.T * d)[0]))
    return sum(ls) / len(ls)


def np_mats(verts):
    """Exact R2 element D (stiffness) and F (mass), as float64 numpy arrays."""
    I3 = sp.eye(3)
    D, F = element.element_matrices(verts, I3, I3)
    Dn = np.array(D.evalf(40), dtype=float)
    Fn = np.array(F.evalf(40), dtype=float)
    return 0.5 * (Dn + Dn.T), 0.5 * (Fn + Fn.T)


def cond_sym(M, drop_null=True):
    """Condition number of a symmetric matrix as max/min nonzero |eigenvalue|.
    The curl-curl D has a 9-dim kernel, so its nonzero conditioning is what
    the factorization actually sees once the mass term lifts the kernel."""
    w = np.abs(np.linalg.eigvalsh(M))
    wmax = w.max()
    if drop_null:
        w = w[w > wmax * 1e-13]
    return wmax / w.min()


# --------------------------------------------------------------------------
# (1) conditioning vs sliver flatness
# --------------------------------------------------------------------------
def part1_blowup():
    print("=" * 74)
    print("(1) element conditioning vs sliver flatness  (canonical R2, unit tensors)")
    print("=" * 74)
    print(f"{'h':>10}{'q=6V/he^3':>14}{'cond(D)':>13}{'cond(F)':>13}{'cond(D+F)':>13}")
    rows = []
    for k in range(0, 8):
        h = sp.Rational(1, 10) ** k
        verts = sliver_tet(h)
        q = float(six_volume(verts) / mean_edge(verts) ** 3)
        D, F = np_mats(verts)
        cD = cond_sym(D, drop_null=True)
        cF = cond_sym(F, drop_null=False)
        cA = cond_sym(D + F, drop_null=False)
        rows.append((float(h), q, cD, cF, cA))
        print(f"{float(h):>10.0e}{q:>14.2e}{cD:>13.3e}{cF:>13.3e}{cA:>13.3e}")

    # empirical scaling exponents (log-log slope of cond vs 1/q)
    q = np.array([r[1] for r in rows])
    cA = np.array([r[4] for r in rows])
    slope = np.polyfit(np.log(1.0 / q[2:]), np.log(cA[2:]), 1)[0]
    print(f"\n  cond(D+F) ~ (1/q)^{slope:.2f}   (q = normalized volume 6V/h_e^3)")
    return rows


# --------------------------------------------------------------------------
# floor threshold: where does the element lose half / all its digits?
# --------------------------------------------------------------------------
def part1b_floor(rows):
    print("\n" + "-" * 74)
    print("floor threshold from the cond(q) trend (u = 2^-52 ≈ 2.2e-16)")
    print("-" * 74)
    u = 2.0 ** -52
    q = np.array([r[1] for r in rows])
    cA = np.array([r[4] for r in rows])
    # fit cond ≈ C / q^p, then invert for the q giving a target condition number
    p, logC = np.polyfit(np.log(1.0 / q[2:]), np.log(cA[2:]), 1)
    C = np.exp(logC)
    def q_for_cond(kappa):
        return (C / kappa) ** (1.0 / p)
    print(f"  fitted cond(q) ≈ {C:.2e} * (1/q)^{p:.2f}")
    for kappa, label in [(1e8, "half precision lost"),
                         (1e12, "safe direct-solve ceiling"),
                         (1.0 / u, "numerically singular")]:
        print(f"    cond = {kappa:.0e} ({label:<28}) at q ≈ {q_for_cond(kappa):.1e}")
    print("  => guard 6V so that q = 6V/h_e^3 stays above ~1e-9 (else the tet is")
    print("     numerically dead); below that, floor it and warn rather than emit NaN.")


# --------------------------------------------------------------------------
# (2) diagonal equilibration: solution-preserving conditioning fix
# --------------------------------------------------------------------------
def jacobi_scale(A):
    d = np.sqrt(np.abs(np.diag(A)))
    S = 1.0 / d
    return S


def ruiz(A, iters=5):
    """Ruiz equilibration: iterate symmetric scaling toward unit-norm rows."""
    n = A.shape[0]
    s = np.ones(n)
    M = A.copy()
    for _ in range(iters):
        r = np.sqrt(np.maximum(np.max(np.abs(M), axis=1), 1e-300))
        d = 1.0 / r
        M = (d[:, None] * M) * d[None, :]
        s *= d
    return s


def part2_equilibration(rows):
    print("\n" + "=" * 74)
    print("(2) symmetric diagonal equilibration  A -> S A S   (S = diag, solution-safe)")
    print("=" * 74)
    # a representative sliver
    h = sp.Rational(1, 1000)
    verts = sliver_tet(h)
    D, F = np_mats(verts)
    A = D + F  # SPD H(curl) Gram matrix; stands in for the system block
    rng = np.random.default_rng(0)
    x_true = rng.standard_normal(20)
    b = A @ x_true

    s_j = jacobi_scale(A)
    A_j = (s_j[:, None] * A) * s_j[None, :]
    s_r = ruiz(A, iters=6)
    A_r = (s_r[:, None] * A) * s_r[None, :]

    print(f"  sliver h={float(h):.0e}:")
    print(f"    cond(A)        = {cond_sym(A):.3e}")
    print(f"    cond(S A S)    = {cond_sym(A_j):.3e}   (Jacobi, 1 step)")
    print(f"    cond(S A S)    = {cond_sym(A_r):.3e}   (Ruiz, 6 steps)")

    # solution preservation: solve scaled system, unscale, compare
    # A x = b  <=>  (S A S)(S^-1 x) = S b ;  recover x = S y
    y = np.linalg.solve(A_j, s_j * b)
    x_rec = s_j * y
    err = np.linalg.norm(x_rec - x_true) / np.linalg.norm(x_true)
    print(f"    ||x_recovered - x_true|| / ||x_true|| = {err:.2e}  (solution preserved)")


# --------------------------------------------------------------------------
# (3) iterative refinement: recover lost digits
# --------------------------------------------------------------------------
def part3_iterative_refinement():
    print("\n" + "=" * 74)
    print("(3) iterative refinement   x_{k+1} = x_k + A^-1 (b - A x_k)")
    print("=" * 74)
    # Mixed-precision refinement converges only when rho = cond(A)*u_factor < 1.
    # Use a MODERATE sliver so a single-precision factorization (u~6e-8) is
    # below that threshold; then refinement recovers full double accuracy.
    h = sp.Rational(1, 100)
    D, F = np_mats(sliver_tet(h))
    A = D + F
    kappa = cond_sym(A)
    u32 = 2.0 ** -23
    rng = np.random.default_rng(1)
    x_true = rng.standard_normal(20)
    b = A @ x_true

    A32 = A.astype(np.float32)                          # the "lossy" factorization
    x = np.linalg.solve(A32, b.astype(np.float32)).astype(np.float64)

    def rel(x):
        return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    print(f"  moderate sliver h={float(h):.0e}: cond(A) = {kappa:.2e}")
    print(f"  contraction rho = cond*u_factor = {kappa*u32:.2e}  ({'<1 converges' if kappa*u32 < 1 else '>1 DIVERGES'})")
    print(f"    step 0 (single-prec solve)   rel err = {rel(x):.2e}")
    for k in range(1, 5):
        r = b - A @ x                                  # residual in working (double) precision
        delta = np.linalg.solve(A32, r.astype(np.float32)).astype(np.float64)
        x = x + delta
        print(f"    step {k} (+1 refinement)      rel err = {rel(x):.2e}")
    print("  => each re-solve reuses the factorization and contracts the error by")
    print("     ~rho, down to the working precision floor. CEILING: when")
    print("     cond*u_factor > 1 (a true sliver, cond > 1/u ~ 4.5e15 in double)")
    print("     refinement DIVERGES — no solver-side trick rescues it; only the")
    print("     volume floor (part 1b) and not meshing slivers do.")


if __name__ == "__main__":
    rows = part1_blowup()
    part1b_floor(rows)
    part2_equilibration(rows)
    part3_iterative_refinement()
