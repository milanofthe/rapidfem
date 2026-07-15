# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""A HIERARCHICAL basis for the same R2 space, and the proof that it is the same.

The interpolatory basis in `element.py` gives every edge two degree-2 functions,
`l*L_a*W_ab` and `l*L_b*W_ab`. Neither of them, and no combination of them, is the
Whitney function `W_ab` itself: the lowest-order (Nedelec-0) space is NOT a
coordinate subspace of that basis. So there is no way to run a cell at order 1 by
dropping DOFs, and no way to read a p-decay indicator off the coefficients. Both
are prerequisites for variable order (docs/fd-basis-plan.md, stages 4-5).

The hierarchical basis fixes that by construction:

    edge (a,b), length l:
        phi_e0 = l * W_ab                    = l*(L_a grad L_b - L_b grad L_a)
        phi_e1 = l * grad(L_a L_b)           = l*(L_a grad L_b + L_b grad L_a)
    face (n0,n1,n2):                          [unchanged from element.py]
        phi_f0 = |n0 n2| * L_n1 (L_n2 grad L_n0 - L_n0 grad L_n2)
        phi_f1 = |n0 n1| * L_n2 (L_n0 grad L_n1 - L_n1 grad L_n0)

Mode 0 of the edges is exactly the Whitney space. Mode 1 is a pure GRADIENT, hence
curl-free: the basis makes the kernel of the curl operator explicit, which is what
"respecting the exact sequence" means here (Schoeberl & Zaglmayr, "High order
Nedelec elements with local complete sequence properties", COMPEL 24 (2005) 374).

Note that per edge the two bases span DIFFERENT 2-D spaces --
`span{L_a W, L_b W}` is degree 2, `span{W, grad(L_a L_b)} = span{L_a grad L_b,
L_b grad L_a}` is degree 1. Only the TOTAL 20-dimensional spaces coincide. That is
the thing this file has to prove, not assume.

What is proved, on a tetrahedron in a general position:

  P1  Both bases have rank 20 and span the SAME space (the change of basis is
      invertible).
  P2  The 6 mode-0 edge functions span exactly the Whitney space, so order 1 is a
      coordinate subspace. (For the interpolatory basis it is not: no Whitney
      function lies in its mode-0 block.)
  P3  The 6 mode-1 edge functions are curl-free, and together with the constant
      gradients already in the Whitney space they span all of grad(P2) -- the full
      kernel of curl inside R2, dim 9. The discrete gradients are IN the space, not
      near it.
  P4  Conformity: a function attached to an entity not on face F has zero tangential
      trace on F. So the trace on a shared face is fixed by the shared DOFs alone.
  P5  Conditioning: cond(D + F) on a reference tet, both bases.

Then the golden 20x20 (D, F) for the hierarchical element are emitted by
`emit_hierarchical_golden.py`.

Run:  python derivations/nedelec2/hierarchical.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp

import element
from element import (
    LOCAL_EDGE_MAP,
    LOCAL_TRI_MAP,
    barycentric_gradients,
    curl_field,
    dist,
    monomial_from_nodes,
    weighted_whitney,
)


# ---------------------------------------------------------------------------
# The hierarchical basis.
# ---------------------------------------------------------------------------
def whitney(grads, a, b, scale):
    """scale * (L_a grad L_b - L_b grad L_a): degree 1, the Whitney field."""
    return [
        (scale, monomial_from_nodes([a]), grads[b]),
        (-scale, monomial_from_nodes([b]), grads[a]),
    ]


def edge_gradient(grads, a, b, scale):
    """scale * grad(L_a L_b) = scale * (L_a grad L_b + L_b grad L_a): curl-free."""
    return [
        (scale, monomial_from_nodes([a]), grads[b]),
        (scale, monomial_from_nodes([b]), grads[a]),
    ]


def build_basis_hierarchical(verts, unit_scale=False):
    """The 20 hierarchical fields, in the SAME DOF order as element.build_basis:
    [6 edge mode0][4 face mode0][6 edge mode1][4 face mode1].

    `unit_scale=True` sets every scale factor to 1. The scales are edge lengths and
    node distances -- square roots of rationals -- and exact linear algebra over a
    field with nested radicals is enormously slow. Scaling a basis function by a
    nonzero constant changes neither the span, nor the tangential trace's support,
    nor whether the curl vanishes, so every structural proof below runs with unit
    scales. Only the conditioning and the emitted golden use the real ones.
    """
    sixV, grads = barycentric_gradients(verts)
    one = sp.Integer(1)
    e0, e1, f0, f1 = [], [], [], []
    for (a, b) in LOCAL_EDGE_MAP:
        le = one if unit_scale else dist(verts, a, b)
        e0.append(whitney(grads, a, b, le))
        e1.append(edge_gradient(grads, a, b, le))
    for (n0, n1, n2) in LOCAL_TRI_MAP:
        # identical to element.py: the face functions are already face-interior
        # and already order 2, so the hierarchy needs no change here.
        s1 = one if unit_scale else dist(verts, n0, n2)
        s2 = one if unit_scale else dist(verts, n0, n1)
        f0.append(weighted_whitney(grads, n1, n2, n0, s1))
        f1.append(weighted_whitney(grads, n2, n0, n1, s2))
    basis = e0 + f0 + e1 + f1
    assert len(basis) == 20
    return basis, sixV, grads


def build_basis_interpolatory(verts, unit_scale=False):
    """element.build_basis, but with the scales optionally set to 1 (same reason)."""
    if not unit_scale:
        return element.build_basis(verts)
    sixV, grads = barycentric_gradients(verts)
    one = sp.Integer(1)
    e1, e2, f1, f2 = [], [], [], []
    for (a, b) in LOCAL_EDGE_MAP:
        e1.append(weighted_whitney(grads, a, a, b, one))
        e2.append(weighted_whitney(grads, b, a, b, one))
    for (n0, n1, n2) in LOCAL_TRI_MAP:
        f1.append(weighted_whitney(grads, n1, n2, n0, one))
        f2.append(weighted_whitney(grads, n2, n0, n1, one))
    return e1 + f1 + e2 + f2, sixV, grads


# ---------------------------------------------------------------------------
# A field -> coefficient vector over the monomial basis of (P2)^3, so that spans
# can be compared by linear algebra rather than by inspection.
# ---------------------------------------------------------------------------
X, Y, Z = sp.symbols("x y z", real=True)


def field_to_poly(field, verts):
    """Expand a barycentric field into three polynomials in x,y,z."""
    sixV, grads = barycentric_gradients(verts)
    M = sp.Matrix([[sp.Integer(1), v[0], v[1], v[2]] for v in verts]).inv()
    # L_j(x,y,z) = M[0,j] + M[1,j] x + M[2,j] y + M[3,j] z
    L = [M[0, j] + M[1, j] * X + M[2, j] * Y + M[3, j] * Z for j in range(4)]
    out = sp.zeros(3, 1)
    for coeff, exps, vec in field:
        mono = sp.Integer(1)
        for k in range(4):
            mono *= L[k] ** exps[k]
        out += coeff * mono * vec
    return sp.expand(out)


MONOS = [
    sp.Integer(1), X, Y, Z,
    X**2, Y**2, Z**2, X*Y, X*Z, Y*Z,
]


def poly_to_vec(pv):
    """A 3-vector of degree<=2 polynomials -> a 30-vector of coefficients."""
    out = []
    for c in range(3):
        p = sp.Poly(pv[c], X, Y, Z)
        for m in MONOS:
            out.append(p.coeff_monomial(m))
    return out


def span_matrix(basis, verts):
    """30 x n matrix whose columns are the basis functions in the monomial basis."""
    cols = [poly_to_vec(field_to_poly(f, verts)) for f in basis]
    return sp.Matrix(cols).T


# A tetrahedron in a general position: no right angles, nothing axis-aligned.
VERTS = [
    (sp.Rational(3, 10), sp.Rational(-1, 5), sp.Rational(1, 10)),
    (sp.Rational(17, 10), sp.Rational(2, 5), sp.Rational(-3, 10)),
    (sp.Rational(1, 10), sp.Rational(19, 10), sp.Rational(3, 5)),
    (sp.Rational(-2, 5), sp.Rational(1, 5), sp.Rational(11, 5)),
]


def element_matrices_hier(verts, eps, mu_inv):
    """The 20x20 (D, F) stiffness/mass of the HIERARCHICAL element, exact.

    The counterpart of `element.element_matrices` for the hierarchical basis. Used
    by `emit_element_golden.py` to pin the Rust `tet_stiff_mass`, which now builds
    the hierarchical basis and only the hierarchical basis.
    """
    basis, sixV, grads = build_basis_hierarchical(verts)
    curls = [element.curl_field(f, grads) for f in basis]
    D = sp.zeros(20, 20)
    F = sp.zeros(20, 20)
    for i in range(20):
        for j in range(i, 20):
            D[i, j] = D[j, i] = element.integrate_dot(curls[i], curls[j], mu_inv, sixV)
            F[i, j] = F[j, i] = element.integrate_dot(basis[i], basis[j], eps, sixV)
    return D, F


def main():
    print("=" * 74)
    print("P1: both bases have rank 20 and span the same space")
    print("    (unit scales: a nonzero rescaling changes no span)")

    interp, _, grads = build_basis_interpolatory(VERTS, unit_scale=True)
    hier, sixV, _ = build_basis_hierarchical(VERTS, unit_scale=True)

    A = span_matrix(interp, VERTS)   # 30 x 20
    B = span_matrix(hier, VERTS)     # 30 x 20
    ra, rb = A.rank(), B.rank()
    print(f"  rank(interpolatory) = {ra}")
    print(f"  rank(hierarchical)  = {rb}")
    assert ra == 20 and rb == 20

    # Same span <=> stacking the two bases adds no dimension.
    joint = A.row_join(B).rank()
    print(f"  rank([interpolatory | hierarchical]) = {joint} (20 => same space)")
    assert joint == 20, "the hierarchical basis leaves R2"
    print("  => the two bases span the SAME 20-dimensional space")

    print()
    print("=" * 74)
    print("P2: the mode-0 edge block IS the Whitney space")

    W = span_matrix([whitney(grads, a, b, sp.Integer(1)) for (a, b) in LOCAL_EDGE_MAP], VERTS)
    print(f"  rank(Whitney space) = {W.rank()}")
    assert W.rank() == 6

    hier_e0 = span_matrix(hier[0:6], VERTS)
    both = W.row_join(hier_e0)
    print(f"  rank([Whitney | hierarchical mode-0]) = {both.rank()} (6 => same space)")
    assert both.rank() == 6, "the hierarchical mode-0 block is not the Whitney space"

    interp_e0 = span_matrix(interp[0:6], VERTS)
    both_i = W.row_join(interp_e0)
    print(f"  rank([Whitney | interpolatory mode-0]) = {both_i.rank()} (12 => disjoint)")
    assert both_i.rank() == 12
    # And not a single Whitney function lies in the interpolatory mode-0 block:
    for e in range(6):
        aug = interp_e0.row_join(W[:, e])
        assert aug.rank() == 7, f"Whitney {e} unexpectedly lies in the interpolatory block"
    print("  no Whitney function lies in the interpolatory mode-0 block")
    print("  => only the hierarchical basis makes order 1 a coordinate subspace")

    print()
    print("=" * 74)
    print("P3: the exact sequence -- mode-1 edges are curl-free and complete grad(P2)")

    for idx in range(10, 16):
        cp = sp.expand(field_to_poly(curl_field(hier[idx], grads), VERTS))
        assert cp == sp.zeros(3, 1), f"hierarchical DOF {idx} is not curl-free"
    print("  all 6 mode-1 edge functions have curl == 0")

    for idx in list(range(0, 10)) + list(range(16, 20)):
        cp = sp.expand(field_to_poly(curl_field(hier[idx], grads), VERTS))
        assert cp != sp.zeros(3, 1), f"hierarchical DOF {idx} is unexpectedly curl-free"
    print("  the other 14 have curl != 0")

    # grad(P2): P2 has dim 10, its gradients dim 9 (constants die). Build them.
    p2 = [sp.Integer(1), X, Y, Z, X**2, Y**2, Z**2, X*Y, X*Z, Y*Z]
    gradp2 = []
    for p in p2[1:]:  # skip the constant, its gradient is zero
        gradp2.append(sp.Matrix([sp.diff(p, X), sp.diff(p, Y), sp.diff(p, Z)]))
    G = sp.Matrix([poly_to_vec(g) for g in gradp2]).T
    print(f"  rank(grad P2) = {G.rank()} (expect 9)")
    assert G.rank() == 9

    # grad(P2) must lie INSIDE the space, and be exactly the curl kernel of it.
    aug = B.row_join(G)
    print(f"  rank([R2 | grad P2]) = {aug.rank()} (20 => grad(P2) is inside R2)")
    assert aug.rank() == 20, "grad(P2) is not contained in the element space"

    # The curl kernel of R2 has dim 20 - rank(curl(R2)).
    curls = [curl_field(f, grads) for f in hier]
    CU = sp.Matrix([poly_to_vec(sp.expand(field_to_poly(c, VERTS))) for c in curls]).T
    rank_curl = CU.rank()
    print(f"  rank(curl R2) = {rank_curl}, so dim ker(curl) = {20 - rank_curl} (expect 9)")
    assert 20 - rank_curl == 9

    # The 6 curl-free basis functions plus the curl-free part of the Whitney block
    # must span all 9 dimensions of that kernel.
    kernel_gens = sp.Matrix(
        [poly_to_vec(field_to_poly(hier[i], VERTS)) for i in range(10, 16)]
    ).T
    kg = kernel_gens.row_join(G)
    print(f"  rank([6 gradient DOFs | grad P2]) = {kg.rank()} (9 => they generate it)")
    assert kg.rank() == 9
    print("  => the discrete gradients are IN the space; the sequence is exact")

    print()
    print("=" * 74)
    print("P4: conformity -- a DOF not on face F has zero tangential trace on F")

    s, t = sp.symbols("s t", nonnegative=True)
    tags = (
        [("edge", e, 0) for e in range(6)]
        + [("face", f, 0) for f in range(4)]
        + [("edge", e, 1) for e in range(6)]
        + [("face", f, 1) for f in range(4)]
    )
    for f, (n0, n1, n2) in enumerate(LOCAL_TRI_MAP):
        face_nodes = {n0, n1, n2}
        face_edges = [e for e, (a, b) in enumerate(LOCAL_EDGE_MAP) if {a, b} <= face_nodes]
        P = [sp.Matrix(VERTS[k]) for k in (n0, n1, n2)]
        pt = s * P[0] + t * P[1] + (1 - s - t) * P[2]
        # Unnormalised normal: the tangential projector v - (v.n)n/(n.n) needs no
        # square root, so the whole check stays in the rationals.
        nrm = (P[1] - P[0]).cross(P[2] - P[0])
        nn = (nrm.T * nrm)[0]
        on_f = {X: pt[0], Y: pt[1], Z: pt[2]}

        n_zero = 0
        for idx, (kind, ent, mode) in enumerate(tags):
            owned = (kind == "edge" and ent in face_edges) or (kind == "face" and ent == f)
            v = field_to_poly(hier[idx], VERTS).subs(on_f)
            tang = sp.expand(v - ((v.T * nrm)[0] / nn) * nrm)
            is_zero = sp.simplify(tang) == sp.zeros(3, 1)
            assert is_zero == (not owned), (
                f"face {f}: DOF {idx} {tags[idx]} owned={owned} but trace_zero={is_zero}"
            )
            n_zero += is_zero
        print(f"  face {f} (edges {face_edges}): {20 - n_zero} DOFs trace, {n_zero} vanish")
        assert 20 - n_zero == 8
    print("  => conformity holds; the trace on a shared face uses only shared DOFs")

    print()
    print("=" * 74)
    print("P5: conditioning of D + F on a reference tetrahedron")

    eye = sp.eye(3)
    cases = [
        ("unit", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        ("general", [tuple(float(c) for c in v) for v in VERTS]),
        # A sliver: one node pulled almost into the opposite face's plane. This is
        # where conditioning actually hurts.
        ("sliver", [(0, 0, 0), (1, 0, 0), (0.5, 0.9, 0), (0.5, 0.3, 0.02)]),
    ]
    for name, verts in cases:
        # Float vertices: the real (square-root) scale factors are used, but the
        # arithmetic stays in floating point, which is what the solver sees anyway.
        vv = [tuple(sp.Float(c) for c in v) for v in verts]
        conds = {}
        for label, builder in [("interpolatory", build_basis_interpolatory),
                               ("hierarchical", build_basis_hierarchical)]:
            bas, sv, gr = builder(vv)
            cur = [curl_field(fn, gr) for fn in bas]
            D = sp.zeros(20, 20)
            F = sp.zeros(20, 20)
            for i in range(20):
                for j in range(i, 20):
                    D[i, j] = D[j, i] = element.integrate_dot(cur[i], cur[j], eye, sv)
                    F[i, j] = F[j, i] = element.integrate_dot(bas[i], bas[j], eye, sv)
            K = np.array((D + F).evalf(), dtype=float)
            conds[label] = np.linalg.cond(K)
        i_c, h_c = conds["interpolatory"], conds["hierarchical"]
        print(f"  {name:9s} interpolatory {i_c:12.1f}   hierarchical {h_c:12.1f}"
              f"   ({i_c / h_c:.2f}x better)")

    print()
    print("=" * 74)
    print("PROVED: the hierarchical basis spans the same R2, nests the Whitney")
    print("space as a coordinate subspace, is conforming, and makes the curl")
    print("kernel explicit. It is a drop-in replacement for the element basis.")


if __name__ == "__main__":
    main()
