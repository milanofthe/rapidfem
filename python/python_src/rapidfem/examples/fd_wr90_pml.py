"""WR-90 waveguide terminated in a PML, single-port matched load.

A PML volume absorbs outgoing waves with vastly less reflection than
a first-order ABC, especially at oblique incidence, the trade is that
PML is a *volumetric* region (a mesh-resolved shell) rather than a
single boundary surface.

Setup: WR-90 with the driven port at z=0 and a PML cap at z=L_INNER.
With the PML tuned correctly, |S11| should be at the numerical floor
across the whole single-mode band, that's the matched-load signature.
"""

# %% Parameters
import math
import numpy as np
import rapidfem as rf

A, B = 22.86e-3, 10.16e-3
L_INNER = 40.0e-3
PML_T   = 15.0e-3

FREQUENCIES = np.linspace(8.0e9, 12.0e9, 21)
MAXH = rf.lambda_maxh(f_max=12.0e9)


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)

inner = g.box(A, B, L_INNER, position=(-A / 2, -B / 2, 0),
              material=rf.Air())

# PML region is also Air (the absorption profile is the PML BC, not the bulk).
pml = g.box(A, B, PML_T, position=(-A / 2, -B / 2, L_INNER),
            material=rf.Air(), maxh=2 * MAXH)

g.fragment(inner, pml)

# Port + BCs. The fragmented interface at z=L_INNER is shared by both volumes;
# we attach the port to the bottom and PEC every "outer" face on either volume.
rf.RectWaveguidePort(inner.faces.min(axis="z"))
rf.PML(pml, direction=(0, 0, 1), inner_face=L_INNER, thickness=PML_T,
       exponent=1.5, delta_max=8.0)

# Everything left on the outer hull → PEC. `.outer` drops the shared inner/pml
# interface (it lives inside the combined bbox); `.unassigned` drops the port
# face we already targeted.
rf.PEC(*inner.faces.outer.unassigned, *pml.faces.outer.unassigned)

rf.show(g)


# %% Mesh
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.ProblemFD(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

s11_dB = [20 * math.log10(max(abs(result.sparams[i, 0, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
print(f"|S11| range across band: {min(s11_dB):.1f} dB to {max(s11_dB):.1f} dB")
print("(a well-tuned PML termination shows |S11| well below -30 dB everywhere)")
