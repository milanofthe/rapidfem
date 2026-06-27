# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Is the existing kernel's element the canonical Nedelec first-kind order 2?

The canonical R2 space (Nedelec 1980) is
    R2 = (P1)^3  (+)  S2,   S2 = { p in (homogeneous P2)^3 : x . p = 0 }, dim 8.
S2 is spanned by X x q for q a homogeneous-linear vector field, X=(x,y,z).

This builds a *raw* polynomial basis of R2 on the unit tet, assembles its
element (D,F) with the same exact integration as element.py, and compares the
basis-independent generalized eigenspectrum to the existing kernel's matrices
(/tmp/emerge_DF.txt). Spectrum match  <=>  the kernel's element is canonical R2.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.linalg import eigh

import element

# Unit tet: x=L2, y=L3, z=L4 (1-based barycentric, v0..v3).
# field term = (coeff, (e1,e2,e3,e4) exps over L, const 3-vector)


def e(c):
    v = [sp.Integer(0)] * 3
    v[c] = sp.Integer(1)
    return sp.Matrix(v)


def p1_fields():
    """12 raw (P1)^3 fields: const and linear per component."""
    fields = []
    for c in range(3):
        fields.append([(sp.Integer(1), (0, 0, 0, 0), e(c))])      # const
        fields.append([(sp.Integer(1), (0, 1, 0, 0), e(c))])      # x = L2
        fields.append([(sp.Integer(1), (0, 0, 1, 0), e(c))])      # y = L3
        fields.append([(sp.Integer(1), (0, 0, 0, 1), e(c))])      # z = L4
    return fields


# position vector X = (x,y,z) as field-vector components in barycentric monomials
X_MONO = [(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]  # (x,y,z) exps


def cross_X_q(q_exps, q_vec):
    """X x (monomial_q * q_vec): X has components x=L2,y=L3,z=L4.

    Returns field terms for the cross product of position vector X with the
    homogeneous-linear field (L^q_exps)*q_vec.
    """
    # X as list of (exps, unit axis)
    Xterms = [(X_MONO[0], 0), (X_MONO[1], 1), (X_MONO[2], 2)]
    out = []
    # q field = monomial(q_exps) along axis given by q_vec (a unit axis)
    qaxis = [i for i in range(3) if q_vec[i] != 0][0]
    levi = {(0, 1): (2, 1), (1, 2): (0, 1), (2, 0): (1, 1),
            (1, 0): (2, -1), (2, 1): (0, -1), (0, 2): (1, -1)}
    for (xexp, xaxis) in Xterms:
        if xaxis == qaxis:
            continue
        comp, sign = levi[(xaxis, qaxis)]
        exps = tuple(xexp[k] + q_exps[k] for k in range(4))
        out.append((sp.Integer(sign), exps, e(comp)))
    return out


def s2_fields():
    """9 candidate X x q fields (rank 8) spanning S2."""
    fields = []
    for q_exp in X_MONO:                # q homogeneous-linear: x,y,z
        for c in range(3):              # along each axis
            fields.append(cross_X_q(q_exp, e(c)))
    return fields


def pick_independent(fields, verts, want):
    """Return a maximal independent subset (by 30-dim polyvec) up to `want`."""
    chosen, cols = [], []
    A = None
    for f in fields:
        v = element.field_to_polyvec(f, verts)
        if A is None:
            test = v
            r_new = 1 if v.norm() != 0 else 0
            if r_new:
                chosen.append(f); cols.append(v); A = v
            continue
        M = sp.Matrix.hstack(A, v)
        if M.rank() > A.rank():
            chosen.append(f); cols.append(v); A = M
        if len(chosen) == want:
            break
    return chosen, A


def load_emerge(path):
    D = F = None
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            vals = np.array([float(x) for x in tok[1:]]).reshape(20, 20)
            if tok[0] == "D":
                D = vals
            elif tok[0] == "F":
                F = vals
    return D, F


def main():
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    cand = p1_fields() + s2_fields()
    basis, A = pick_independent(cand, verts, 20)
    print(f"canonical R2 raw basis: {len(basis)} independent fields, "
          f"polyspace rank {A.rank()}")
    assert len(basis) == 20, "failed to extract 20-dim R2 basis"

    I3 = sp.eye(3)
    sixV, grads = element.barycentric_gradients(verts)
    curls = [element.curl_field(f, grads) for f in basis]
    D = sp.zeros(20, 20)
    F = sp.zeros(20, 20)
    for i in range(20):
        for j in range(i, 20):
            D[i, j] = D[j, i] = element.integrate_dot(curls[i], curls[j], I3, sixV)
            F[i, j] = F[j, i] = element.integrate_dot(basis[i], basis[j], I3, sixV)
    Dr = np.array(D.evalf(), dtype=float)
    Fr = np.array(F.evalf(), dtype=float)

    De, Fe = load_emerge("/tmp/emerge_DF.txt")
    sym = lambda M: 0.5 * (M + M.T)
    Dr, Fr, De, Fe = map(sym, (Dr, Fr, De, Fe))

    wr = np.sort(eigh(Dr, Fr, eigvals_only=True))
    we = np.sort(eigh(De, Fe, eigvals_only=True))
    print("\ngeneralized eigenvalues  D x = lambda F x")
    print(f"{'canonical R2':>20}{'kernel (EMerge)':>20}{'abs diff':>12}")
    maxd = 0.0
    for a, b in zip(wr, we):
        maxd = max(maxd, abs(a - b))
        print(f"{a:20.10e}{b:20.10e}{abs(a-b):12.2e}")
    rel = maxd / max(1.0, np.max(np.abs(we)))
    print(f"\nmax |dlambda| = {maxd:.3e}   rel = {rel:.3e}")
    print("=> EMerge IS canonical R2." if rel < 1e-9
          else "=> EMerge is NOT canonical R2 (different element).")


if __name__ == "__main__":
    main()
