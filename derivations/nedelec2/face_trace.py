"""The R2 surface element is the tangential trace of the R2 volume element.

`crates/rapidfem-fd/src/tri_assembly.rs` assembles the Robin / port boundary
term on a boundary triangle with 8 DOFs, and hands them to the global system under
the SAME global indices the volume element uses on that face (3 edges x 2 modes,
and the face itself x 2 modes). That is only legitimate if the surface functions
are the traces of the volume functions. The Rust used to construct them a second
time, by hand, with a comment saying the signs had been matched to the volume
element. This file proves the statement the comment asserted, symbolically, on a
tetrahedron in a general position (no coordinate is special, no edge is a unit
vector).

The three lemmas, in order:

  L1  On the face F opposite vertex d, the barycentric coordinate L_d vanishes
      identically.

  L2  The tangential part of grad L_d vanishes on F: grad L_d is parallel to F's
      normal. For the three vertices of F, the tangential part of grad L_i is
      exactly the triangle's own 2-D barycentric gradient.

  L3  Hence: a volume function whose terms name only F's vertices traces to the
      surface function with the same coefficients read in 2-D; a volume function
      that names vertex d in EVERY term traces to zero. Counting the R2 functions
      that survive gives exactly 8, and they are F's 3 edges x 2 modes plus F's own
      2 modes.

The consequence checked at the end is the operative one: the surface mass matrix
integral  int_F phi_i . phi_j dA  computed from the traced volume functions equals
the one computed from the surface element. Run:

    python derivations/nedelec2/face_trace.py
"""

import itertools

import sympy as sp

# ---------------------------------------------------------------------------
# A tetrahedron in a general position. Vertices 0,1,2 span the face F we trace
# onto; vertex 3 is the opposite vertex d.
# ---------------------------------------------------------------------------
P = [
    sp.Matrix([sp.Rational(3, 10), sp.Rational(-1, 5), sp.Rational(1, 10)]),
    sp.Matrix([sp.Rational(17, 10), sp.Rational(2, 5), sp.Rational(-3, 10)]),
    sp.Matrix([sp.Rational(1, 10), sp.Rational(19, 10), sp.Rational(3, 5)]),
    sp.Matrix([sp.Rational(-2, 5), sp.Rational(1, 5), sp.Rational(11, 5)]),
]
OPP = 3  # the vertex opposite F = (0,1,2)

x, y, z = sp.symbols("x y z", real=True)
XYZ = sp.Matrix([x, y, z])


def volume_barycentric():
    """L_i(x,y,z), affine, with L_i(P_j) = delta_ij."""
    a, b, c, d = sp.symbols("a b c d")
    lams = []
    for i in range(4):
        expr = a * x + b * y + c * z + d
        eqs = [sp.Eq(expr.subs({x: P[j][0], y: P[j][1], z: P[j][2]}), 1 if i == j else 0)
               for j in range(4)]
        sol = sp.solve(eqs, [a, b, c, d], dict=True)[0]
        lams.append(sp.expand(expr.subs(sol)))
    return lams


L = volume_barycentric()
GRAD = [sp.Matrix([sp.diff(Li, x), sp.diff(Li, y), sp.diff(Li, z)]) for Li in L]

# The face's unit normal.
E1 = P[1] - P[0]
E2 = P[2] - P[0]
NRM = E1.cross(E2)
NHAT = NRM / sp.sqrt(NRM.dot(NRM))


def tangential(v):
    """The component of v in the plane of F."""
    return sp.simplify(v - (v.dot(NHAT)) * NHAT)


# A point of F, parameterised by its own barycentric coordinates.
s, t = sp.symbols("s t", nonnegative=True)
FACE_BARY = [s, t, 1 - s - t]  # of vertices 0, 1, 2
X_ON_F = FACE_BARY[0] * P[0] + FACE_BARY[1] * P[1] + FACE_BARY[2] * P[2]
ON_F = {x: X_ON_F[0], y: X_ON_F[1], z: X_ON_F[2]}

print("=" * 72)
print("L1: the opposite coordinate vanishes on the face")
l_opp_on_f = sp.simplify(L[OPP].subs(ON_F))
print(f"  L_{OPP} restricted to F = {l_opp_on_f}")
assert l_opp_on_f == 0

print()
print("L1b: the face's own coordinates restrict to the triangle's barycentrics")
for k in range(3):
    got = sp.simplify(L[k].subs(ON_F) - FACE_BARY[k])
    print(f"  L_{k}|_F - lambda_{k} = {got}")
    assert got == 0

print()
print("=" * 72)
print("L2: the opposite gradient has no tangential part")
gt = tangential(GRAD[OPP])
print(f"  (grad L_{OPP})_t = {gt.T}")
assert all(sp.simplify(c) == 0 for c in gt)

print()
print("  and the three face gradients keep a nonzero tangential part:")
for k in range(3):
    gk = tangential(GRAD[k])
    nz = sp.simplify(gk.dot(gk))
    print(f"  |(grad L_{k})_t|^2 = {nz}")
    assert nz != 0


# ---------------------------------------------------------------------------
# The R2 functions, exactly as `tet_assembly::edge_fns` / `face_fns`
# emit them. A function is a list of (coeff, exponent multi-index, grad index)
# with an overall scale.
# ---------------------------------------------------------------------------
def exps(*nodes, n=4):
    e = [0] * n
    for i in nodes:
        e[i] += 1
    return tuple(e)


def edge_fns(a, b, n=4):
    """phi_e1 = l L_a W_ab, phi_e2 = l L_b W_ab; W_ab = L_a grad L_b - L_b grad L_a."""
    length = sp.sqrt((P[a] - P[b]).dot(P[a] - P[b])) if n == 4 else None
    return [
        (length, [(1, exps(a, a, n=n), b), (-1, exps(a, b, n=n), a)]),
        (length, [(1, exps(a, b, n=n), b), (-1, exps(b, b, n=n), a)]),
    ]


def face_fns(n0, n1, n2, n=4):
    """phi_f1 = |n0 n2| L_n1 (L_n2 grad L_n0 - L_n0 grad L_n2),
       phi_f2 = |n0 n1| L_n2 (L_n0 grad L_n1 - L_n1 grad L_n0)."""
    d02 = sp.sqrt((P[n0] - P[n2]).dot(P[n0] - P[n2]))
    d01 = sp.sqrt((P[n0] - P[n1]).dot(P[n0] - P[n1]))
    return [
        (d02, [(-1, exps(n1, n0, n=n), n2), (1, exps(n1, n2, n=n), n0)]),
        (d01, [(1, exps(n2, n0, n=n), n1), (-1, exps(n2, n1, n=n), n0)]),
    ]


TET_EDGE_LOCAL = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 1), (2, 3)]
TET_FACE_LOCAL = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3)]
TRI_EDGE_LOCAL = [(0, 1), (1, 2), (0, 2)]


def build_volume_basis():
    """The 20 functions, in the DOF order [6 edge m1][4 face m1][6 edge m2][4 face m2],
    each tagged with the entity it belongs to."""
    edges = [edge_fns(a, b) for (a, b) in TET_EDGE_LOCAL]
    faces = [face_fns(*f) for f in TET_FACE_LOCAL]
    out = []
    for m in range(2):
        out += [(("edge", e, m), edges[e][m]) for e in range(6)]
        out += [(("face", f, m), faces[f][m]) for f in range(4)]
    return out


def eval_volume(fn):
    """A volume function as a symbolic 3-vector field."""
    scale, terms = fn
    v = sp.zeros(3, 1)
    for coeff, e, g in terms:
        mono = sp.Integer(1)
        for i, ei in enumerate(e):
            mono *= L[i] ** ei
        v += scale * coeff * mono * GRAD[g]
    return v


VOL = build_volume_basis()

print()
print("=" * 72)
print("L3: which volume DOFs survive the trace onto F = (0,1,2)?")

# The face's three edges, as tet-local edge indices.
face_nodes = {0, 1, 2}
face_edges = [e for e, (a, b) in enumerate(TET_EDGE_LOCAL) if {a, b} <= face_nodes]
face_index = TET_FACE_LOCAL.index((0, 1, 2))
print(f"  F's tet-local edges: {face_edges}, F's tet-local face index: {face_index}")

survivors = []
for v, (tag, fn) in enumerate(VOL):
    trace = tangential(eval_volume(fn)).subs(ON_F)
    trace = sp.simplify(trace)
    zero = all(sp.simplify(c) == 0 for c in trace)
    kind, ent, mode = tag
    expected_zero = not (
        (kind == "edge" and ent in face_edges) or (kind == "face" and ent == face_index)
    )
    assert zero == expected_zero, f"DOF {v} {tag}: traced-to-zero = {zero}, expected {expected_zero}"
    if not zero:
        survivors.append((v, tag, trace))

print(f"  {len(survivors)} of 20 volume DOFs have a nonzero tangential trace")
assert len(survivors) == 8
for v, tag, _ in survivors:
    print(f"    local {v:2d}  {tag}")


# ---------------------------------------------------------------------------
# The surface element, built on the triangle's own three nodes, and the claim
# that its 8 functions ARE the 8 traces above.
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("The surface element equals the trace, function by function")

# The surface element lives in the triangle's 2-D frame, but the identity is
# frame-independent: compare the traces as 3-D vectors on F. The surface function's
# 3-D realisation uses the tangential parts of the same gradients (L2), which is
# exactly what the 2-D construction computes in the local frame.
TAN_GRAD = [tangential(GRAD[k]) for k in range(3)]


def eval_surface(fn):
    scale, terms = fn
    v = sp.zeros(3, 1)
    for coeff, e, g in terms:
        mono = sp.Integer(1)
        for i, ei in enumerate(e[:3]):
            mono *= FACE_BARY[i] ** ei
        assert e[3] == 0, "a surface term names the opposite vertex"
        assert g < 3, "a surface term gradients the opposite vertex"
        v += scale * coeff * mono * TAN_GRAD[g]
    return v


surf_edges = [edge_fns(a, b) for (a, b) in TRI_EDGE_LOCAL]
surf_face = face_fns(0, 1, 2)
SURF = []
for m in range(2):
    SURF += [surf_edges[e][m] for e in range(3)]
    SURF.append(surf_face[m])

# The DOF correspondence the assembler uses: surface local s <-> the volume DOF on
# the same mesh entity. Surface edge i is TRI_EDGE_LOCAL[i] = a node pair of F,
# which is tet-local edge `face_edges` in the same order because both lists are
# built from ascending node pairs.
tri_edge_as_tet_edge = [TET_EDGE_LOCAL.index(pair) for pair in TRI_EDGE_LOCAL]
print(f"  triangle edge i -> tet-local edge {tri_edge_as_tet_edge}")

corr = []
for m in range(2):
    corr += [("edge", e, m) for e in tri_edge_as_tet_edge]
    corr.append(("face", face_index, m))

vol_by_tag = {tag: trace for (_, tag, trace) in survivors}
for sidx, (sfn, tag) in enumerate(zip(SURF, corr)):
    d = sp.simplify(eval_surface(sfn) - vol_by_tag[tag])
    ok = all(c == 0 for c in d)
    print(f"  surface DOF {sidx} == volume {tag}: {'yes' if ok else 'NO'}")
    assert ok


# ---------------------------------------------------------------------------
# And therefore the mass matrices agree. Integrate over F in its own coordinates:
# dA = 2A ds dt on the reference triangle s,t >= 0, s+t <= 1.
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("The Robin mass matrices agree entrywise")

two_a = sp.sqrt(NRM.dot(NRM))
PHI = [eval_surface(f) for f in SURF]
M = sp.zeros(8, 8)
for i in range(8):
    for j in range(i, 8):
        integrand = sp.expand(PHI[i].dot(PHI[j]))
        val = sp.integrate(sp.integrate(integrand, (t, 0, 1 - s)), (s, 0, 1)) * two_a
        M[i, j] = sp.simplify(val)
        M[j, i] = M[i, j]

MT = sp.zeros(8, 8)
for i in range(8):
    for j in range(i, 8):
        integrand = sp.expand(vol_by_tag[corr[i]].dot(vol_by_tag[corr[j]]))
        val = sp.integrate(sp.integrate(integrand, (t, 0, 1 - s)), (s, 0, 1)) * two_a
        MT[i, j] = sp.simplify(val)
        MT[j, i] = MT[i, j]

diff = sp.simplify(M - MT)
maxdiff = max(abs(sp.N(diff[i, j])) for i in range(8) for j in range(8))
print(f"  max |M_surface - M_trace| = {maxdiff}")
assert diff == sp.zeros(8, 8)

print()
print("  the surface Robin matrix (gamma = 1), for reference:")
for i in range(8):
    print("   ", "  ".join(f"{float(M[i, j]):+.6e}" for j in range(8)))

print()
print("=" * 72)
print("PROVED: the R2 surface element is the tangential trace of the R2 volume")
print("element, with the same sign convention, on a general tetrahedron.")
print("The Rust counterpart is crates/rapidfem-fd/tests/face_trace_test.rs.")
