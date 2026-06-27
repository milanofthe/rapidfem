"""Import a STEP solid and drive it as a first-class FEM region.

Shows the external-CAD flow: ``g.load("part.step")`` brings an OpenCASCADE
solid into the *same* kernel the primitives use, so it returns a normal
``GeoObject`` you attach materials, ports, and boolean ops to exactly like a
``g.box(...)``.

Notebook-style flow:

    STEP file  ->  load as GeoObject  ->  Materials + Ports  ->  Mesh  ->  Problem

The STEP file here is generated on the fly with gmsh so the script runs
end-to-end; in practice ``STEP_PATH`` is just your CAD export.
"""

# %% Make a stand-in STEP part (replace with your own export)
import os
import tempfile

import numpy as np
import gmsh

import rapidfem as rf

A, B, L = 22.86e-3, 10.16e-3, 30.0e-3        # WR-90 width, height, length [m]
FREQUENCIES = np.linspace(8.0e9, 12.0e9, 21)

# Author a WR-90-sized box and write it to STEP. gmsh tags the file in
# millimetres, so rapidfem's default unit="M" reads it back in metres.
STEP_PATH = os.path.join(tempfile.gettempdir(), "rapidfem_demo_part.step")
if not gmsh.isInitialized():
    gmsh.initialize()
gmsh.model.add("demo_part")
gmsh.model.occ.addBox(0, 0, 0, A * 1e3, B * 1e3, L * 1e3)  # in mm
gmsh.model.occ.synchronize()
gmsh.write(STEP_PATH)
gmsh.model.remove()


# %% Load the STEP solid as the air region
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12.0e9))
part = g.load(STEP_PATH, material=rf.Air())   # GeoObject, fully composable

# Pick the two end faces as ports, everything else is PEC wall, the same
# face selectors that work on a primitive work on the imported solid.
rf.RectWaveguidePort(part.faces.min(axis="z"))
rf.RectWaveguidePort(part.faces.max(axis="z"))
rf.PEC(*part.faces.unassigned)

# Compose with primitives / booleans just like any GeoObject, e.g.:
#   post = g.cylinder(radius=1e-3, height=B, position=(0, -B/2, L/2))
#   g.cut(part, post)        # subtract a tuning post from the imported solid

rf.show(g)


# %% Mesh + sweep
g.mesh()
prob = rf.Problem(g)
result = prob.sweep(FREQUENCIES)
rf.show(result)

print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
print(f"|S11| at f0: {abs(result.sparams[0, 0, 0]):.4g}")
print(f"|S21| at f0: {abs(result.sparams[0, 1, 0]):.4g}")
