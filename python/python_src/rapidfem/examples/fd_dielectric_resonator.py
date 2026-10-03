"""Dielectric resonator filter, eigenmode analysis.

A high-εᵣ ceramic puck sits on a low-εᵣ alumina support inside a metal
cavity. The PEC walls turn this into a resonator with bound eigenmodes;
the puck concentrates the field via its high εᵣ.
"""

# %% Parameters
import numpy as np
import rapidfem as rf

mm = 1e-3
inch = 25.4 * mm

W = 2.0 * inch
S = 2.03 * inch

D_SUP, L_SUP = 0.56 * inch, 0.80 * inch
ER_SUP = 10.0

D_RES, L_RES = 1.176 * inch, 0.481 * inch
ER_RES = 34.0

F_TARGET = 2.0e9
N_MODES = 5


# %% Geometry + Materials
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=3.0e9))

air = g.box(W, W, S, position=(-W / 2, -W / 2, 0), material=rf.Air())
support = g.cylinder(radius=D_SUP / 2, height=L_SUP,
                     material=rf.Dielectric(er=ER_SUP),
                     maxh=rf.lambda_maxh(f_max=3.0e9, er_max=ER_SUP))
resonator = g.cylinder(radius=D_RES / 2, height=L_RES, position=(0, 0, L_SUP),
                       material=rf.Dielectric(er=ER_RES),
                       maxh=rf.lambda_maxh(f_max=3.0e9, er_max=ER_RES))

g.fragment(air, support, resonator)

# Only the 6 axis-aligned cavity walls are PEC. Selecting min/max along each
# axis avoids tagging the cylinder-air interface faces that fragment exposes:
# the air-puck and air-support interfaces must stay un-walled.
rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
       air.faces.min(axis="y"), air.faces.max(axis="y"),
       air.faces.min(axis="z"), air.faces.max(axis="z"))

rf.show(g)


# %% Mesh
g.mesh()
rf.show(g)


# %% Eigenmode
prob = rf.ProblemFD(g)
modes = prob.eigenmode(target_frequency=F_TARGET, n_modes=N_MODES)
rf.show(prob)
rf.show(modes)

print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
print(f"Found {len(modes)} modes near {F_TARGET/1e9:.2f} GHz:")
for i, m in enumerate(modes):
    f_real = m.frequency_hz / 1e9
    f_imag = m.frequency_imag_hz / 1e9
    q = m.q_factor
    q_str = f"Q={q:.1f}" if np.isfinite(q) else "Q=∞"
    print(f"  mode {i+1}: f = {f_real:.4f} GHz   (imag {f_imag:+.2e} GHz, {q_str})")
