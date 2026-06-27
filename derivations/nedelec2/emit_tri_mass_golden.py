# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Emit the Rust golden test for the R2 surface (boundary-triangle) mass matrix.

Derives the 8×8 Robin surface mass matrix  γ·∫ φ_i·φ_j dA  on a boundary
triangle SYMBOLICALLY, by an independent re-construction of the exact tangential
R2 surface basis that `crates/rapidfem-fd/src/tri_assembly_r2.rs::ned2_tri_stiff`
implements, integrated exactly via the barycentric area identity

    ∫_tri L_a^p L_b^q L_c^r dA = 2·p!·q!·r!/(p+q+r+2)! · Area .

The emitted cargo test calls the real Rust `ned2_tri_stiff` for several test
triangles and γ values (real and complex) and asserts the 8×8 result matches
this symbolic golden entrywise to ~1e-10 relative.

Surface element (8 DOF = 3 edges × 2 modes + 1 face × 2 modes), with the exact
DOF order, weights, gradient pairings and signs read off the Rust source:

  TRI_EDGE_MAP = [[0,1],[1,2],[0,2]]
  edge e=(a,b) m1 (weight a): φ = ℓ·(  L_a·L_a·∇L_b − L_a·L_b·∇L_a )
  edge e=(a,b) m2 (weight b): φ = ℓ·(  L_b·L_a·∇L_b − L_b·L_b·∇L_a )
  face m1: φ = |0,2|·( −L_1·L_0·∇L_2 + L_1·L_2·∇L_0 )
  face m2: φ = |0,1|·(  L_2·L_0·∇L_1 − L_2·L_1·∇L_0 )
  DOF order: [edge0 e1 e2 (m1)], face·m1, [edge0 e1 e2 (m2)], face·m2.
"""
from __future__ import annotations

import os

import sympy as sp

I = sp.I

# Surface local edge order, must match tri_assembly_r2.rs::TRI_EDGE_MAP.
TRI_EDGE_MAP = [(0, 1), (1, 2), (0, 2)]


def tri_local_2d(verts3):
    """Map 3-D triangle vertices to the same local 2-D frame as tri_local_cs:
    vertex 0 at origin, edge 0→1 along x̂, n̂ = ê1×ê2 as ẑ. Returns (xs, ys)."""
    v0 = sp.Matrix(verts3[0])
    v1 = sp.Matrix(verts3[1])
    v2 = sp.Matrix(verts3[2])
    e1 = v1 - v0
    e2 = v2 - v0
    xhat = e1 / sp.sqrt((e1.T * e1)[0])
    zvec = e1.cross(e2)
    zhat = zvec / sp.sqrt((zvec.T * zvec)[0])
    yhat = zhat.cross(xhat)
    xs = [sp.Integer(0)] * 3
    ys = [sp.Integer(0)] * 3
    for k, v in enumerate((v0, v1, v2)):
        d = v - v0
        xs[k] = sp.simplify((xhat.T * d)[0])
        ys[k] = sp.simplify((yhat.T * d)[0])
    return xs, ys


def bary_grads_2d(xs, ys):
    """2-D barycentric gradients ∇L_i and signed 2A, exactly as bary_grads_2d."""
    x1, x2, x3 = xs
    y1, y2, y3 = ys
    two_a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    b = [y2 - y3, y3 - y1, y1 - y2]
    c = [x3 - x2, x1 - x3, x2 - x1]
    grads = [sp.Matrix([b[i] / two_a, c[i] / two_a]) for i in range(3)]
    return grads, two_a


def node_dist(xs, ys, i, j):
    return sp.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)


# A surface basis function: (scale, [(coeff, (m0, m1), grad), ...]).
def build_surface_basis(xs, ys):
    d = lambda i, j: node_dist(xs, ys, i, j)

    def e(a, b, weight):
        # ℓ·( L_weight·L_a·∇L_b − L_weight·L_b·∇L_a )
        return (d(a, b), [(1, (weight, a), b), (-1, (weight, b), a)])

    e0, e1, e2 = TRI_EDGE_MAP
    f1 = (d(0, 2), [(-1, (1, 0), 2), (1, (1, 2), 0)])
    f2 = (d(0, 1), [(1, (2, 0), 1), (-1, (2, 1), 0)])
    return [
        e(e0[0], e0[1], e0[0]),   # 0: edge0 m1
        e(e1[0], e1[1], e1[0]),   # 1: edge1 m1
        e(e2[0], e2[1], e2[0]),   # 2: edge2 m1
        f1,                        # 3: face m1
        e(e0[0], e0[1], e0[1]),   # 4: edge0 m2
        e(e1[0], e1[1], e1[1]),   # 5: edge1 m2
        e(e2[0], e2[1], e2[1]),   # 6: edge2 m2
        f2,                        # 7: face m2
    ]


def area_integral(p, q, r, s, area):
    """∫_tri L_p L_q L_r L_s dA over the triangle (0-based node indices 0,1,2).

    = 2·Area·∏ m_i!/(Σ m_i + 2)!  with m_i the multiplicity of node i."""
    mult = [0, 0, 0]
    for idx in (p, q, r, s):
        mult[idx] += 1
    num = 1
    for mi in mult:
        num *= sp.factorial(mi)
    total = sum(mult)
    return 2 * area * sp.Rational(num, sp.factorial(total + 2))


def tri_mass_matrix(verts3, gamma):
    """8×8 symbolic γ·∫ φ_i·φ_j dA, mirroring ned2_tri_stiff exactly."""
    xs, ys = tri_local_2d(verts3)
    grads, two_a = bary_grads_2d(xs, ys)
    area = sp.Abs(two_a) / 2
    fns = build_surface_basis(xs, ys)
    M = sp.zeros(8, 8)
    for i in range(8):
        sc_i, terms_i = fns[i]
        for j in range(i, 8):
            sc_j, terms_j = fns[j]
            acc = sp.Integer(0)
            for (ci, mi, gi) in terms_i:
                for (cj, mj, gj) in terms_j:
                    g = (grads[gi].T * grads[gj])[0]
                    intg = area_integral(mi[0], mi[1], mj[0], mj[1], area)
                    acc += ci * cj * g * intg
            val = sp.simplify(gamma * sc_i * sc_j * acc)
            M[i, j] = M[j, i] = val
    return M


# (name, verts3, gamma)
CASES = [
    ("unit_real", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], sp.Integer(1)),
    ("unit_complex", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], sp.Rational(3, 10) + sp.Rational(6, 5) * I),
    ("scaled_real", [(0, 0, 0), (2, 0, 0), (0, 3, 0)], sp.Rational(5, 2)),
    (
        "skew_complex",
        [(0, 0, 0), (2, 0, 1), (sp.Rational(1, 2), 1, sp.Rational(3, 10))],
        sp.Rational(3, 2) - sp.Rational(7, 10) * I,
    ),
]


def c64(z):
    z = complex(sp.N(z, 30))
    return f"C64::new({z.real:.17e}, {z.imag:.17e})"


def f64(v):
    return f"{float(sp.N(v, 30)):.17e}"


def emit_verts(name, verts3):
    rows = ", ".join("[" + ", ".join(f64(v[k]) for k in range(3)) + "]" for v in verts3)
    return f"const V_{name}: [[f64; 3]; 3] = [{rows}];\n"


def emit_matrix(name, M):
    rows = []
    for i in range(8):
        entries = ", ".join(c64(M[i, j]) for j in range(8))
        rows.append(f"    [{entries}],")
    body = "\n".join(rows)
    return f"const M_{name}: [[C64; 8]; 8] = [\n{body}\n];\n"


HEADER = """\
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// GENERATED by derivations/nedelec2/emit_tri_mass_golden.py — do not edit.
// Golden 8×8 surface Robin mass matrix  γ·∫ φ_i·φ_j dA  on a boundary triangle,
// from an independent symbolic reconstruction of the R2 tangential surface
// basis (exact barycentric area integration). Pins ned2_tri_stiff entrywise.

use num_complex::Complex64 as C64;
use rapidfem_fd::coefficients::AreaCoeffCache;
use rapidfem_fd::tri_assembly_r2::ned2_tri_stiff;

fn maxdiff(a: &[[C64; 8]; 8], b: &[[C64; 8]; 8]) -> f64 {
    let mut m = 0.0_f64;
    let mut scale = 1e-300_f64;
    for i in 0..8 { for j in 0..8 {
        scale = scale.max(b[i][j].norm());
        m = m.max((a[i][j] - b[i][j]).norm());
    }}
    m / scale
}
"""


def emit_case(name, verts3, gamma):
    M = tri_mass_matrix(verts3, gamma)
    g = c64(gamma)
    out = f"\n// ===== case {name} (gamma = {complex(sp.N(gamma, 20))}) =====\n"
    out += emit_verts(name, verts3)
    out += emit_matrix(name, M)
    out += f"""
#[test]
fn ned2_tri_stiff_matches_derivation_{name}() {{
    let ac = AreaCoeffCache::new();
    let got = ned2_tri_stiff(&V_{name}, {g}, &ac);
    let err = maxdiff(&got, &M_{name});
    eprintln!("{name}: max rel err {{:.2e}}", err);
    assert!(err < 1e-10, "mismatch ({name}): {{:.2e}}", err);
}}
"""
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    out_path = os.path.join(
        repo, "crates", "rapidfem-fd", "tests", "tri_mass_golden_test.rs"
    )
    body = HEADER
    for name, verts3, gamma in CASES:
        body += emit_case(name, verts3, gamma)
        print(f"emitted case {name}")
    with open(out_path, "w") as fh:
        fh.write(body)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
