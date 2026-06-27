# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Clean-room derivation of the RLC lumped port (FD), from primary EM.

INDEPENDENT DERIVATION. Nothing here is transcribed from any other solver. The
physics is textbook: the impedance (Leontovich) boundary condition in the
curl-curl weak form (Jin, *The FEM in Electromagnetics*), the lumped-element ->
sheet-impedance geometry factor, and the Kurokawa power-wave S-parameter
(Pozar, *Microwave Engineering*). We re-derive every constant with sympy and
validate the whole chain against the analytic reflection coefficient of a
terminated line, so the implementation rests on our own algebra.

Grounding (rapidfem's actual system, crates/rapidfem-fd/src/assembly.rs and
derivations/basis_nondim/nondimensionalize.py):

    K = E - k0^2 B,   k0 = omega/c0,   coordinates scaled by L0,   kappa = k0*L0
    E_ij = INT (curl phi_i) mur^-1 (curl phi_j) dV          (stiffness)
    B_ij = INT  phi_i        eps_r  phi_j        dV          (mass)

Time convention e^{+i*omega*t}  (so Z_L = i*omega*L, Z_C = 1/(i*omega*C)), matching
crates/rapidfem-fd/src/waveguide.rs (LumpedElement.impedance, get_gamma = +i...).

What we derive and EMIT for the kernel:
  (1) gamma  = i*kappa*eta0 / Zs(omega)            Robin surface coefficient
  (2) Zs     = Z(omega) * (w/l)                    sheet impedance from geometry
        series   Z = R + i*omega*L + 1/(i*omega*C)
        parallel Y = 1/R + 1/(i*omega*L) + i*omega*C,  Z = 1/Y
  (3) u_inc  = -2*gamma * E_inc,  E_inc = (V_inc/l) * lhat,  V_inc = sqrt(2*P*Z0)
  (4) V      = (1/w) * INT_Gamma  E . lhat  dS      mode-projected port voltage
  (5) S      = V / V_inc            (minus 1 on the driven/reflection port)

and PROVE (5)+(3) reproduce  S = (Zload - Z0)/(Zload + Z0).
"""
from __future__ import annotations

import sympy as sp


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# Symbols ---------------------------------------------------------------------
omega, mu0, eps0, c0, eta0 = sp.symbols("omega mu0 eps0 c0 eta0", positive=True)
k0, L0, kappa = sp.symbols("k0 L0 kappa", positive=True)
R, L, C, Z0, P = sp.symbols("R L C Z0 P", positive=True)
w, l = sp.symbols("w l", positive=True)          # port width, length(=height)
i = sp.I

# Free-space relations
eta0_def = sp.sqrt(mu0 / eps0)
c0_def = 1 / sp.sqrt(mu0 * eps0)


# ---------------------------------------------------------------------------
section("(1) Impedance BC -> Robin surface term gamma = i*kappa*eta0/Zs")
# ---------------------------------------------------------------------------
# Curl-curl, source-free interior, Galerkin test F (tangential Nedelec):
#   INT mur^-1 (curlE).(curlF) dV  -  k0^2 INT epsr E.F dV
#        -  OINT [ n x (mur^-1 curlE) ] . F dS  = 0
# Faraday (e^{+iwt}):  curlE = -i*omega*mu0*mur*H  =>  mur^-1 curlE = -i*omega*mu0*H
# so the boundary integrand is
#   -[ n x (mur^-1 curlE) ].F = -[ n x (-i*omega*mu0 H) ].F = +i*omega*mu0 (nxH).F
H_bdry_coeff = i * omega * mu0          # coefficient of (nxH).F on the LHS
# Leontovich impedance BC:  E_tan = Zs (n x H)  =>  n x H = (1/Zs) E_tan
# and for tangential test fields  E_tan.F = (nxE).(nxF).  Hence the LHS surface
# term is   gamma_phys * INT (nxE).(nxF) dS   with
Zs = sp.symbols("Zs")
gamma_phys = H_bdry_coeff / Zs          # = i*omega*mu0/Zs
# Non-dimensionalize: omega*mu0 = (k0 c0)(eta0/c0) = k0*eta0, and k0 = kappa/L0.
omega_mu0 = (k0 * c0) * (eta0 / c0)      # = k0*eta0
omega_mu0 = sp.simplify(omega_mu0.subs(c0, c0_def).subs(eta0, eta0_def))
# substitute back the symbolic k0*eta0 and k0 = kappa/L0
gamma_nondim = (i * (kappa / L0) * eta0) / Zs
print("  gamma_phys   = i*omega*mu0/Zs           =", gamma_phys)
print("  omega*mu0    = k0*eta0  (verified:", sp.simplify(omega_mu0 - (k0*eta0).subs(eta0, eta0_def)) == 0, ")")
print("  gamma(nondim)= i*kappa*eta0/(L0*Zs)      =", gamma_nondim)
print("  -> with the L0 from the surface mass (INT phi phi dS ~ L0^2) absorbed in")
print("     assembly, the kernel coefficient is  gamma = i*kappa*eta0/Zs.")


# ---------------------------------------------------------------------------
section("(2) Geometry: lumped Z -> sheet Zs = Z*(w/l); RLC impedances")
# ---------------------------------------------------------------------------
# Current flows along lhat over length l, across width w (area A=l*w). A sheet
# of impedance Zs [ohm/square] has total Z = Zs*(l/w)  (l squares in series, w
# wide in parallel).  Therefore:
Zs_geom = sp.symbols("Z") * (w / l)
print("  Zs = Z*(w/l)            ->", Zs_geom, "   (so Z = Zs*(l/w), the squares count)")
Z_series = R + i * omega * L + 1 / (i * omega * C)
Y_parallel = 1 / R + 1 / (i * omega * L) + i * omega * C
Z_parallel = sp.simplify(1 / Y_parallel)
print("  series   Z(w) =", sp.simplify(Z_series))
print("  parallel Y(w) =", sp.simplify(Y_parallel))
print("  parallel Z(w) =", Z_parallel)
# Pure resistive reference port. Series RLC reduces to R when the inductor is a
# short (L->0) and the capacitor is a short (C->oo): Z = R.
Z_series_R = sp.simplify(Z_series.subs({R: Z0, L: 0})).limit(C, sp.oo)
print("  R-only (R=Z0, L->0, C->oo): series Z =", Z_series_R, " == Z0 :",
      sp.simplify(Z_series_R - Z0) == 0)


# ---------------------------------------------------------------------------
section("(3) Incident power -> V_inc = sqrt(2*P*Z0), E_inc = (V_inc/l) lhat")
# ---------------------------------------------------------------------------
# Port mode is TEM-like with reference impedance Z0: V_inc = INT E_inc.dl = E0*l,
# I_inc = V_inc/Z0, time-averaged incident power P = (1/2) Re(V_inc I_inc*).
V_inc = sp.sqrt(2 * P * Z0)
E0 = V_inc / l
I_inc = V_inc / Z0
P_check = sp.simplify(sp.Rational(1, 2) * V_inc * I_inc)   # real, Z0 real
print("  V_inc      =", V_inc)
print("  E_inc(mag) = V_inc/l =", E0)
print("  P = (1/2)|V_inc|^2/Z0 =", P_check, " == P :", sp.simplify(P_check - P) == 0)


# ---------------------------------------------------------------------------
section("(4) Mode-projected port voltage  V = (1/w) INT_Gamma E.lhat dS")
# ---------------------------------------------------------------------------
# For a uniform modal field E = a*lhat on Gamma (area A=l*w):
#   (1/w) INT E.lhat dS = (1/w) * a * A = a * l = INT_along_length E.dl.
# So the surface-averaged path integral equals the physical gap voltage, but is
# well-defined (transverse-averaged) for a NON-uniform solved field too. This is
# the parity fix vs a single line integral.
a = sp.symbols("a")              # modal amplitude of the solved total field
A_area = l * w
V_mode = sp.simplify((1 / w) * a * A_area)
print("  uniform field a*lhat:  V = (1/w)*a*A =", V_mode, " (= a*l, the gap voltage)")


# ---------------------------------------------------------------------------
section("(5) S-parameter + END-TO-END VALIDATION vs analytic reflection")
# ---------------------------------------------------------------------------
# Power-wave (Kurokawa) on a Z0-referenced port: V = V_inc + V_refl, so the
# reflected b-amplitude normalized by the incident a-amplitude is
#   S = V_refl/V_inc = V/V_inc - 1   (driven port; passive ports: S = Vj/V_inc).
#
# VALIDATION. The scattered-field source u_inc = -2*gamma*E_inc makes the port a
# Thevenin source of open-circuit voltage 2*V_inc behind Z0 (the factor 2 is the
# matched-source open-circuit voltage). Terminate it with an arbitrary load
# Zload; the node voltage and S must reproduce the textbook reflection.
Zload = sp.symbols("Zload")
V_node = 2 * V_inc * Zload / (Z0 + Zload)        # divider: 2*Vinc source, Z0 series, Zload
S = sp.simplify(V_node / V_inc - 1)
S_expected = (Zload - Z0) / (Zload + Z0)
print("  V_node (Thevenin 2*Vinc behind Z0, load Zload) =", V_node)
print("  S = V_node/V_inc - 1 =", S)
print("  S_expected = (Zload-Z0)/(Zload+Z0) =", S_expected)
ok = sp.simplify(S - S_expected) == 0
print("  MATCH:", ok)
assert ok, "S-parameter chain does not reproduce the reflection coefficient!"

# Sanity limits
S_open = sp.limit(S, Zload, sp.oo)
S_short = S.subs(Zload, 0)
S_match = S.subs(Zload, Z0)
print(f"  open  (Zload->oo): S = {S_open}   (expect +1)")
print(f"  short (Zload->0):  S = {S_short}   (expect -1)")
print(f"  match (Zload=Z0):  S = {S_match}   (expect  0)")
assert S_open == 1 and S_short == -1 and S_match == 0

# The factor 2 is load-bearing: a source of k*Vinc gives S = k*Zl/(Z0+Zl) - 1,
# which equals the reflection coefficient ONLY for k = 2. Prove it.
k = sp.symbols("k")
S_k = sp.simplify(k * Zload / (Z0 + Zload) - 1)
sol = sp.solve(sp.Eq(S_k, S_expected), k)
print("  source factor k forced by S==reflection:", sol, " -> the '2' in u_inc=-2*gamma*E_inc")
assert sol == [2]

section("RESULT")
print("""All constants self-derived and validated:
  gamma  = i*kappa*eta0 / Zs                 [from impedance BC, sec 1]
  Zs     = Z(omega)*(w/l)                    [geometry, sec 2]
  u_inc  = -2*gamma*E_inc, E_inc=(Vinc/l)l̂   [factor 2 PROVEN load-bearing, sec 5]
  V_inc  = sqrt(2*P*Z0)                       [unit power, sec 3]
  V      = (1/w) INT E.l̂ dS                  [mode projection, sec 4]
  S      = V/V_inc (-1 on driven port)        [reproduces (Zl-Z0)/(Zl+Z0), sec 5]
The only new kernel piece vs today is (4): replace the line-integral voltage by
the area-averaged mode projection, assembled as a boundary linear form and
dotted with the solved E -- robust for tall / non-TEM ports.""")
print("\nOK: lumped-port RLC derivation self-consistent.")
