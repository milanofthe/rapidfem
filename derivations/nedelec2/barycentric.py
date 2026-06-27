# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
#
# Clean-room symbolic derivation. Independent of any third-party source:
# the simplex integration identity is derived here from scratch by direct
# symbolic integration over the reference simplices, then matched against
# its classical closed form (Eisenberg & Malvern, "On finite element
# integration in natural coordinates", IJNME 7 (1973) 574-575).
"""Barycentric (natural-coordinate) integration coefficients.

Two pure-rational tables drive the Nedelec-2 element assembly:

  volume_coeff(a,b,c,d) = (1/6V) * integral_tet  L_a L_b L_c L_d  dV
  area_coeff(a,b,c,d)   = (1/A)  * integral_tri  L_a L_b L_c L_d  dA

where indices in {1,2,3,4} select a barycentric coordinate and index 0 is
"unused" (contributes a factor 1). m_i is the multiplicity of vertex i
among the arguments. Both coefficients are dimensionless and mesh
independent.

This module derives them two independent ways and asserts agreement:
  (A) direct symbolic integration over the reference simplex (sympy), and
  (B) the closed factorial form  prod(m_i!) / (sum(m_i)+K)!  (K=3 vol, 2 area).
Agreement across the whole index range is the clean-room proof that the
hard-coded Rust formula reproduces the *mathematics*, not anyone's code.
"""
from __future__ import annotations

import json
from itertools import product
from math import factorial

import sympy as sp


def _reference_tet_integral(m):
    """Symbolically integrate L1^m1 L2^m2 L3^m3 L4^m4 over the reference tet.

    Reference tet: vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1), so
    L2=x, L3=y, L4=z, L1=1-x-y-z, and 6V=1 (V=1/6). Returns a sympy Rational.
    """
    x, y, z = sp.symbols("x y z", nonnegative=True)
    L = {1: 1 - x - y - z, 2: x, 3: y, 4: z}
    integrand = sp.Integer(1)
    for i in (1, 2, 3, 4):
        integrand *= L[i] ** m[i]
    # nested integral over the standard simplex
    inner = sp.integrate(integrand, (z, 0, 1 - x - y))
    inner = sp.integrate(inner, (y, 0, 1 - x))
    vol_integral = sp.integrate(inner, (x, 0, 1))
    # volume_coeff = integral / (6V); 6V = 1 here
    return sp.nsimplify(vol_integral)


def _reference_tri_integral(m):
    """Symbolically integrate L1^m1 L2^m2 L3^m3 over the reference triangle.

    Reference tri: vertices (0,0),(1,0),(0,1), so L2=x, L3=y, L1=1-x-y,
    A=1/2. Returns area_coeff = integral / A as a sympy Rational.
    """
    x, y = sp.symbols("x y", nonnegative=True)
    L = {1: 1 - x - y, 2: x, 3: y}
    integrand = sp.Integer(1)
    for i in (1, 2, 3):
        integrand *= L[i] ** m[i]
    inner = sp.integrate(integrand, (y, 0, 1 - x))
    area_integral = sp.integrate(inner, (x, 0, 1))
    A = sp.Rational(1, 2)
    return sp.nsimplify(area_integral / A)


def _multiplicities(a, b, c, d, n_vertices):
    """Count how often each vertex index 1..n_vertices appears; 0 is unused."""
    m = {i: 0 for i in range(1, n_vertices + 1)}
    for idx in (a, b, c, d):
        if idx != 0:
            if idx > n_vertices:
                # index out of simplex range -> not representable
                return None
            m[idx] += 1
    return m


def volume_coeff_closed(a, b, c, d):
    """Closed factorial form: prod(m_i!) / (sum(m_i)+3)!  (tet)."""
    m = _multiplicities(a, b, c, d, 4)
    num = 1
    for mi in m.values():
        num *= factorial(mi)
    total = sum(m.values())
    return sp.Rational(num, factorial(total + 3))


def area_coeff_closed(a, b, c, d):
    """Closed factorial form: 2 * prod(m_i!) / (sum(m_i)+2)!  (triangle)."""
    m = _multiplicities(a, b, c, d, 3)
    if m is None:
        return None
    num = 1
    for mi in m.values():
        num *= factorial(mi)
    total = sum(m.values())
    return 2 * sp.Rational(num, factorial(total + 2))


def derive_and_verify():
    """Cross-check symbolic integration vs the closed form over 0..4^4.

    Returns (vol_table, area_table) of exact rationals keyed by (a,b,c,d).
    Raises AssertionError on any mismatch.
    """
    vol_table = {}
    area_table = {}
    for a, b, c, d in product(range(5), repeat=4):
        # --- volume (tet): vertices 1..4 ---
        m = _multiplicities(a, b, c, d, 4)
        sym = _reference_tet_integral(m)
        closed = volume_coeff_closed(a, b, c, d)
        assert sym == closed, f"vol mismatch at {(a,b,c,d)}: {sym} != {closed}"
        vol_table[(a, b, c, d)] = closed

        # --- area (tri): only indices 1..3 live; 4 is out of the triangle ---
        if 4 not in (a, b, c, d):
            m3 = _multiplicities(a, b, c, d, 3)
            sym3 = _reference_tri_integral(m3)
            closed3 = area_coeff_closed(a, b, c, d)
            assert sym3 == closed3, f"area mismatch at {(a,b,c,d)}: {sym3} != {closed3}"
            area_table[(a, b, c, d)] = closed3
    return vol_table, area_table


def export_golden(path):
    """Write a flat golden table of f64 values for the Rust verification test."""
    vol_table, area_table = derive_and_verify()
    out = {
        "volume": [
            {"idx": list(k), "value": float(v)} for k, v in sorted(vol_table.items())
        ],
        "area": [
            {"idx": list(k), "value": float(v)} for k, v in sorted(area_table.items())
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    vol_table, area_table = derive_and_verify()
    print(f"verified {len(vol_table)} volume + {len(area_table)} area coefficients")
    # spot checks against the documented anchors
    assert volume_coeff_closed(0, 0, 0, 0) == sp.Rational(1, 6)
    assert volume_coeff_closed(1, 1, 0, 0) == sp.Rational(2, 120)
    assert volume_coeff_closed(1, 2, 3, 4) == sp.Rational(1, 5040)
    assert area_coeff_closed(0, 0, 0, 0) == 1
    assert area_coeff_closed(1, 1, 0, 0) == sp.Rational(1, 6)
    print("anchors OK: vol(0000)=1/6 vol(1100)=1/60 vol(1234)=1/5040 "
          "area(0000)=1 area(1100)=1/6")
