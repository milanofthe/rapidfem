"""Import an STL solid and drive it as a first-class FEM region.

Shows the external-geometry flow: ``g.load("part.stl")`` turns a closed
surface into a solid (split into smooth faces at its creases), so it
returns a normal ``GeoObject`` you attach materials, ports and boolean ops
to exactly like a ``g.box(...)``. ``unit`` reads the file's coordinates,
``rotation`` and ``position`` place the part.

Notebook-style flow:

    STL file  ->  load as GeoObject  ->  Materials + Ports  ->  Mesh  ->  Problem

The STL file here is written on the fly so the script runs end-to-end; in
practice ``STL_PATH`` is just your CAD export.
"""

# %% Make a stand-in STL part (replace with your own export)
import os
import tempfile

import numpy as np

import rapidfem as rf

A, B, L = 22.86e-3, 10.16e-3, 30.0e-3        # WR-90 width, height, length [m]
FREQUENCIES = np.linspace(8.0e9, 12.0e9, 21)

# A WR-90-sized box in millimetres, two outward triangles per side.
STL_PATH = os.path.join(tempfile.gettempdir(), "rapidfem_demo_part.stl")
corners = np.array([[x, y, z] for z in (0, L) for y in (0, B) for x in (0, A)]) * 1e3
quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
with open(STL_PATH, "w") as f:
    f.write("solid part\n")
    for a, b, c, d in quads:
        for tri in ((a, b, c), (a, c, d)):
            p = corners[list(tri)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            f.write(f"facet normal {' '.join(map(str, n / np.linalg.norm(n)))}\nouter loop\n")
            for v in p:
                f.write(f"vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("endloop\nendfacet\n")
    f.write("endsolid part\n")


# %% Load the STL solid as the air region
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12.0e9))
part = g.load(STL_PATH, unit="MM", material=rf.Air())   # GeoObject, fully composable

# Pick the two end faces as ports, everything else is PEC wall, the same
# face selectors that work on a primitive work on the imported solid.
rf.RectWaveguidePort(part.faces.min(axis="z"))
rf.RectWaveguidePort(part.faces.max(axis="z"))
rf.PEC(*part.faces.unassigned)

# Compose with primitives / booleans just like any GeoObject, e.g.:
#   post = g.cylinder(radius=1e-3, height=B, position=(A / 2, 0, L / 2), axis=(0, 1, 0))
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
