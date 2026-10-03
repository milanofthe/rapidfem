"""Coaxial impedance step (50Ω → 75Ω), quarter-wave matching section."""

# %% Parameters
import numpy as np
import rapidfem as rf

RI_1, RO_1 = 1.50e-3, 3.45e-3   # Section 1: 50 Ω air coax
RI_2, RO_2 = 0.99e-3, 3.45e-3   # Section 2: 75 Ω air coax
L1, L2 = 15.0e-3, 15.0e-3

FREQUENCIES = np.linspace(1.0e9, 10.0e9, 31)
MAXH = rf.lambda_maxh(f_max=10.0e9)


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)

# Outer air dielectric spans both sections; the inner conductors are
# subtracted via fragment, so the inner-conductor surface becomes a PEC face.
air = g.cylinder(radius=RO_1, height=L1 + L2, position=(0, 0, 0),
                 material=rf.Air())
inner_a = g.cylinder(radius=RI_1, height=L1, position=(0, 0, 0),
                     material=rf.Air())
inner_b = g.cylinder(radius=RI_2, height=L2, position=(0, 0, L1),
                     material=rf.Air())

g.fragment(air, inner_a, inner_b)

# Coax ports at the two flat ends of the outer cylinder.
rf.CoaxPort(air.faces.min(axis="z"), ri=RI_1, ro=RO_1, origin=(0, 0, 0))
rf.CoaxPort(air.faces.max(axis="z"), ri=RI_2, ro=RO_2, origin=(0, 0, L1 + L2))

# Everything else (inner-conductor surfaces + outer wall) is PEC.
rf.PEC(*air.faces.unassigned)

rf.show(g)


# %% Mesh
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.ProblemFD(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
print(f"|S11| at {FREQUENCIES[0]/1e9:.1f} GHz:  {abs(result.sparams[0, 0, 0]):.4f}")
print(f"|S11| at {FREQUENCIES[-1]/1e9:.1f} GHz: {abs(result.sparams[-1, 0, 0]):.4f}")
