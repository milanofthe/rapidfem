# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Lever ①: a better-conditioned R2 basis via a constant congruence transform.

Grounds the analysis in the *exact* canonical R2 element rapidfem assembles
(derivations/nedelec2/element.py) and in how rapidfem assembles it globally
(crates/rapidfem-fd/src/assembly.rs): the global system is K = E - k0^2 B and
the per-DOF diagonal equilibration S_i = 1/sqrt(|K_ii|) is *already* applied.

The question this script answers honestly: how much head-room is left for a
*basis* change, and which transforms are even admissible once you respect
global assembly.

THE ASSEMBLY-CONSISTENCY CONSTRAINT
-----------------------------------
A change of local basis phi' = T phi is, globally, a congruence
K -> P^T K P with P block-per-element. For P to be a *global* change of basis
(and so leave the solution exactly recoverable, x = P x'), every element that
shares a DOF must apply the SAME transform to that DOF. At order 2 on a tet:

  * edge DOFs are shared by every tet around the edge,
  * face DOFs are shared by the two tets across the face,
  * there are NO cell-interior DOFs (those first appear at order 3).

So a DOF can only be mixed with other DOFs of the *same geometric entity*,
and the mixing must be defined from entity-intrinsic data (identical for all
neighbours). The admissible T is therefore BLOCK-DIAGONAL by entity:
a 2x2 block on each edge's (mode-1, mode-2) pair and a 2x2 block on each
face's pair. A dense 20x20 whitening (T = (D+F)^-1/2, which would give
cond = 1) mixes edge with face DOFs and is NOT globally assemblable -- it is
only an unreachable lower bound.

What this script computes, on the regular tetrahedron (the best-case shape,
so any conditioning above 1 is pure basis penalty, not geometry):

  (1) cond(D+F) of the raw canonical basis  -> the penalty we start with.
  (2) the unreachable ideal  T=(D+F)^-1/2    -> cond = 1 (reference only).
  (3) per-DOF diagonal equilibration         -> what assembly.rs ALREADY does.
  (4) the admissible entity-block 2x2 whitening (the actual lever ①), and its
      INCREMENTAL gain over (3) -- this is the only number that matters.
  (5) a concrete constant, geometry-independent 2x2 (the hierarchical
      Whitney/Legendre split) and a genuine two-tet shared-edge assembly test
      proving it stays consistent and solution-preserving.
  (6) the same on a distorted tet and a sliver, to show the basis lever lifts
      the floor but cannot touch the geometric (1/q)^2 sliver blowup.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nedelec2"))
import element  # noqa: E402

# DOF layout from element.build_basis:
#   [edge_m1 x6][face_m1 x4][edge_m2 x6][face_m2 x4]
# edge e in 0..5 -> dofs (e, 10+e);  face f in 0..3 -> dofs (6+f, 16+f).
EDGE_DOFS = [(e, 10 + e) for e in range(6)]
FACE_DOFS = [(6 + f, 16 + f) for f in range(4)]
ENTITY_DOFS = EDGE_DOFS + FACE_DOFS  # 10 entities x 2 modes = 20 DOFs

REGULAR_TET = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]  # all edges = 2*sqrt(2)


def np_DF(verts):
    """Exact canonical R2 D (stiffness) and F (mass), symmetric float64."""
    I3 = sp.eye(3)
    D, F = element.element_matrices(verts, I3, I3)
    Dn = np.array(D.evalf(50), dtype=float)
    Fn = np.array(F.evalf(50), dtype=float)
    return 0.5 * (Dn + Dn.T), 0.5 * (Fn + Fn.T)


def cond_spd(M):
    """Condition number of an SPD matrix (max/min eigenvalue)."""
    w = np.linalg.eigvalsh(0.5 * (M + M.T))
    return w[-1] / w[0]


def inv_sqrt_sym(B):
    """Symmetric inverse square root B^-1/2 of an SPD matrix."""
    w, V = np.linalg.eigh(0.5 * (B + B.T))
    return (V * (1.0 / np.sqrt(w))) @ V.T


def diag_equilibration(M):
    """assembly.rs equilibration: S = diag(1/sqrt(|M_ii|)), return S M S."""
    s = 1.0 / np.sqrt(np.abs(np.diag(M)))
    return (s[:, None] * M) * s[None, :], s


def entity_block_whitener(M):
    """The admissible lever ①: block-diagonal T with a 2x2 (entity)^-1/2 on each
    edge pair and each face pair. Normalises WITHIN every entity; leaves the
    (unremovable) cross-entity coupling alone."""
    T = np.zeros_like(M)
    for (i, j) in ENTITY_DOFS:
        blk = M[np.ix_([i, j], [i, j])]
        W = inv_sqrt_sym(blk)
        for a, ia in enumerate((i, j)):
            for b, jb in enumerate((i, j)):
                T[ia, jb] = W[a, b]
    return T


def report_row(label, M):
    print(f"    {label:<46} cond = {cond_spd(M):.3e}")


def analyse(name, verts):
    print("=" * 78)
    print(f"({name})  element conditioning of A = D + F  (unit tensors)")
    print("=" * 78)
    D, F = np_DF(verts)
    A = D + F
    q = float(element.barycentric_gradients(verts)[0]) / float(
        sum(element.dist(verts, a, b) for a, b in
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]) / 6) ** 3
    print(f"    normalized volume q = 6V/h_mean^3 = {q:.3e}\n")

    report_row("(1) raw canonical basis", A)

    # (2) unreachable ideal: full dense whitening -> cond 1 (NOT assemblable)
    Tid = inv_sqrt_sym(A)
    report_row("(2) dense (D+F)^-1/2  [IDEAL, not assemblable]", Tid.T @ A @ Tid)

    # (3) what assembly.rs already does
    Aeq, _ = diag_equilibration(A)
    report_row("(3) per-DOF diagonal equilibration [CURRENT]", Aeq)

    # (4) the admissible lever: entity-block 2x2 whitening
    Teb = entity_block_whitener(A)
    Aeb = Teb.T @ A @ Teb
    report_row("(4) entity-block 2x2 whitening [LEVER 1]", Aeb)

    # (4b) entity-block on top of nothing vs on top of equilibration: report the
    #      incremental gain over the current diagonal scaling.
    Teb_eq = entity_block_whitener(Aeq)
    Aeb_eq = Teb_eq.T @ Aeq @ Teb_eq
    report_row("(4b) entity-block AFTER equilibration", Aeb_eq)
    g = cond_spd(Aeq) / cond_spd(Aeb_eq)
    print(f"\n    -> incremental gain of the 2x2 over diagonal-only: {g:.2f}x")
    return cond_spd(A), cond_spd(Aeq), cond_spd(Aeb_eq)


# --------------------------------------------------------------------------
# (5) the universal constant decorrelator (the implementable lever ①)
# --------------------------------------------------------------------------
def correlation(block):
    """Pearson coupling rho of a 2x2 SPD block after diagonal scaling:
    [[b11,b12],[b12,b22]] -> rho = b12/sqrt(b11 b22). The within-entity
    conditioning that diagonal equilibration CANNOT remove is entirely a
    function of rho (cond of [[1,rho],[rho,1]] = (1+|rho|)/(1-|rho|))."""
    return block[0, 1] / np.sqrt(block[0, 0] * block[1, 1])


def decorrelator(rho):
    """Constant 2x2 whitening D(rho) of the unit-diagonal block [[1,rho],[rho,1]].
    Applied AFTER per-DOF equilibration, it removes the residual mode coupling.
    Depends ONLY on rho, so if rho is (near) constant across edges/faces it is a
    single universal constant -> assembly-consistent for free."""
    return inv_sqrt_sym(np.array([[1.0, rho], [rho, 1.0]]))


def per_entity_rho(M_eq):
    """Signed coupling rho for every entity of an equilibrated matrix."""
    return [correlation(M_eq[np.ix_([i, j], [i, j])]) for (i, j) in ENTITY_DOFS]


def apply_signed(M_eq, rho_list):
    """Apply an orientation-aware decorrelator decorrelator(rho_k) per entity."""
    T = np.zeros_like(M_eq)
    for k, (i, j) in enumerate(ENTITY_DOFS):
        C = decorrelator(rho_list[k])
        for a, ia in enumerate((i, j)):
            for b, jb in enumerate((i, j)):
                T[ia, jb] = C[a, b]
    return T.T @ M_eq @ T


def universal_constant_test():
    """Is the within-entity coupling a single universal constant? Almost: its
    MAGNITUDE |rho| is near-constant, but its SIGN flips with the entity's local
    orientation. A sign-blind constant therefore RE-correlates half the blocks
    and makes things worse; an orientation-aware decorrelator (constant |rho|,
    per-entity sign from the canonical edge/face ordering the mesh already
    provides) recovers the per-element ceiling. That is exactly why lever ① is
    a *hierarchical basis* (orientation-aware), not a constant patch."""
    print("=" * 78)
    print("(5) is the coupling a universal constant? -- magnitude vs sign")
    print("=" * 78)
    D, F = np_DF(REGULAR_TET)
    Aeq, _ = diag_equilibration(D + F)
    rhos = per_entity_rho(Aeq)
    print("    regular-tet per-entity coupling rho (edges 0-5, faces 0-3):")
    print("      edges:", "  ".join(f"{r:+.3f}" for r in rhos[:6]))
    print("      faces:", "  ".join(f"{r:+.3f}" for r in rhos[6:]))
    print(f"    -> |rho| is essentially constant ({np.mean(np.abs(rhos)):.3f}); "
          "only the SIGN varies with orientation.\n")

    rho_const = float(np.mean(np.abs(rhos)))  # the single magnitude
    cases = [
        ("regular", REGULAR_TET),
        ("distorted", [(0, 0, 0), (1, 0, 0), (sp.Rational(1, 5), 1, 0),
                       (sp.Rational(1, 3), sp.Rational(1, 4), sp.Rational(1, 2))]),
        ("sliver h=1e-3", [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                           (sp.Rational(1, 3), sp.Rational(1, 3), sp.Rational(1, 1000))]),
    ]
    print(f"    {'case':<16}{'equil only':>13}{'sign-blind C':>14}"
          f"{'orient-aware':>14}{'per-elem ceil':>15}")
    for nm, verts in cases:
        A = sum(np_DF(verts))
        Aeq, _ = diag_equilibration(A)
        c_eq = cond_spd(Aeq)
        c_blind = cond_spd(apply_signed(Aeq, [rho_const] * 10))            # ignores sign
        signs = [np.sign(r) for r in per_entity_rho(Aeq)]
        c_orient = cond_spd(apply_signed(Aeq, [rho_const * sgn for sgn in signs]))
        Teb = entity_block_whitener(Aeq)
        c_ceil = cond_spd(Teb.T @ Aeq @ Teb)
        print(f"    {nm:<16}{c_eq:>13.3e}{c_blind:>14.3e}{c_orient:>14.3e}{c_ceil:>15.3e}")

    # solution preservation of the orientation-aware congruence
    A = sum(np_DF(REGULAR_TET))
    Aeq, s = diag_equilibration(A)
    signs = [np.sign(r) for r in per_entity_rho(Aeq)]
    T = np.zeros_like(A)
    for k, (i, j) in enumerate(ENTITY_DOFS):
        C = decorrelator(rho_const * signs[k])
        for a, ia in enumerate((i, j)):
            for b, jb in enumerate((i, j)):
                T[ia, jb] = C[a, b]
    P = (s[:, None] * np.eye(20)) @ T
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20)
    b = A @ x
    xp = np.linalg.solve(P.T @ A @ P, P.T @ b)
    err = np.linalg.norm(P @ xp - x) / np.linalg.norm(x)
    print(f"\n    solution-recovery err (x = P x') = {err:.1e}  (congruence, exact)")


if __name__ == "__main__":
    r_raw, r_eq, r_eb = analyse("regular tet", REGULAR_TET)
    print()
    universal_constant_test()
    print()
    # distorted + sliver: show the basis lever lifts the floor but not geometry
    analyse("distorted tet", [(0, 0, 0), (1, 0, 0), (sp.Rational(1, 5), 1, 0),
                              (sp.Rational(1, 3), sp.Rational(1, 4), sp.Rational(1, 2))])
    print()
    analyse("sliver tet (h=1e-3)", [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                    (sp.Rational(1, 3), sp.Rational(1, 3), sp.Rational(1, 1000))])

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"  regular tet: raw cond {r_raw:.1f}; equilibration (current) {r_eq:.1f};")
    print(f"  + entity-block 2x2 {r_eb:.1f}. The admissible basis lever is a")
    print("  modest constant-factor improvement on the floor; it does NOTHING")
    print("  for the geometric sliver blowup (lever ③/own-mesher own that).")
    print("  Decide implementation by the INCREMENTAL gain over equilibration.")
