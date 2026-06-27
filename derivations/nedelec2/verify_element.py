# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Basis-independent equivalence check of the derived Nedelec-2 element.

Two bases span the same finite-element space iff the generalized eigenvalues
of the element pencil (D, F) agree (congruence invariance). D is the
curl-curl stiffness, F the mass; F is SPD. If the spectra of the independent
sympy element and the existing kernel match, the elements are physically
identical and the per-entry matrix difference is a pure basis artifact.

Reads /tmp/emerge_DF.txt (produced by tests/_dump_unit_tet.rs) and compares
against element.py on the unit tet with identity tensors.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.linalg import eigh

import element


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
    I3 = sp.eye(3)
    Ds, Fs = element.element_matrices(verts, I3, I3)
    Dm = np.array(Ds.evalf(), dtype=float)
    Fm = np.array(Fs.evalf(), dtype=float)

    De, Fe = load_emerge("/tmp/emerge_DF.txt")

    # symmetrize tiny numerical asymmetry
    sym = lambda M: 0.5 * (M + M.T)
    Dm, Fm, De, Fe = map(sym, (Dm, Fm, De, Fe))

    wt = eigh(Dm, Fm, eigvals_only=True)   # derived element
    we = eigh(De, Fe, eigvals_only=True)   # existing kernel
    wt = np.sort(wt)
    we = np.sort(we)

    print("generalized eigenvalues  D x = lambda F x  (sorted)")
    print(f"{'derived (sympy)':>22}   {'kernel (EMerge)':>22}   {'abs diff':>10}")
    max_abs = 0.0
    for a, b in zip(wt, we):
        max_abs = max(max_abs, abs(a - b))
        print(f"{a:22.12e}   {b:22.12e}   {abs(a-b):10.2e}")
    print(f"\nmax |Δλ| = {max_abs:.3e}")
    # scale-aware tolerance: spectrum spans ~0..O(60)
    rel = max_abs / max(1.0, np.max(np.abs(we)))
    print(f"max |Δλ| / ||λ||_inf = {rel:.3e}")
    if rel < 1e-10:
        print("\n=> SAME finite-element space: physically identical element.")
    else:
        print("\n=> DIFFERENT space: elements are not equivalent.")
    # null space (gradient) dimension = #(near-zero eigenvalues)
    nz = int(np.sum(np.abs(we) < 1e-9))
    print(f"curl null-space dim (gradient functions): derived "
          f"{int(np.sum(np.abs(wt) < 1e-9))}, kernel {nz}")

    # ---- conditioning (basis-dependent, unlike the spectrum above) ----
    print("\n--- element-matrix conditioning (unit tet, identity tensors) ---")

    def spd_cond(M):
        w = np.linalg.eigvalsh(M)
        w = w[w > w.max() * 1e-14]
        return w.max() / w.min()

    def sym_cond(M):
        # full conditioning incl. sign; report extreme-magnitude ratio
        w = np.abs(np.linalg.eigvalsh(M))
        w = w[w > w.max() * 1e-14]
        return w.max() / w.min()

    print(f"{'metric':<28}{'derived':>14}{'kernel':>14}")
    print(f"{'cond(F) mass SPD':<28}{spd_cond(Fm):>14.3e}{spd_cond(Fe):>14.3e}")
    print(f"{'cond(D+F) shifted SPD':<28}"
          f"{spd_cond(Dm+Fm):>14.3e}{spd_cond(De+Fe):>14.3e}")
    print(f"{'cond(D) curl (nonzero)':<28}"
          f"{sym_cond(Dm):>14.3e}{sym_cond(De):>14.3e}")
    # scaled (diagonal-normalized) mass conditioning — what a Jacobi
    # preconditioner would see
    def jacobi_cond(M):
        d = np.sqrt(np.abs(np.diag(M)))
        S = M / np.outer(d, d)
        return spd_cond(S)
    print(f"{'cond(F) Jacobi-scaled':<28}"
          f"{jacobi_cond(Fm):>14.3e}{jacobi_cond(Fe):>14.3e}")


if __name__ == "__main__":
    main()
