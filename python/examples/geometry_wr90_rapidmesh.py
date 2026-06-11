"""WR-90 straight waveguide meshed by RAPIDMESH, solved by
rapidfem via Simulation.from_arrays — no gmsh, no .msh bytes.

Gates (the same as rapidfem's gmsh-based geometry_wr90.py):
  max |S11| < 0.01, |S21| in [0.99, 1.01] across 9-11 GHz.
"""

import numpy as np

import rapidfem
import rapidmesh as rm

a, b, L = 22.86e-3, 10.16e-3, 30e-3

# ---- rapidmesh: box + coincident port sheets on the end faces
g = rm.Geometry(maxh=3e-3)
g.box(a, b, L)
g.xy_plate(a, b, position=(0, 0, 0), tag=2)  # port 1 at z=0
g.xy_plate(a, b, position=(0, 0, L), tag=3)  # port 2 at z=L
mesh = g.mesh()
print("rapidmesh:", mesh)

# ---- tag plumbing: untagged boundary faces become PEC (tag 1)
tags = mesh.face_tags.astype(np.int32).copy()
boundary = (mesh.face_regions == 0).any(axis=1)
tags[boundary & (tags == 0)] = 1
keep = tags > 0
tris = mesh.faces[keep].astype(np.int64)
tri_tags = tags[keep]
tet_tags = mesh.tet_regions.astype(np.int32)
print(
    f"faces: {int((tri_tags == 1).sum())} pec, "
    f"{int((tri_tags == 2).sum())} port1, {int((tri_tags == 3).sum())} port2"
)

config = f"""
[mesh]
file = "(rapidmesh)"

[frequency]
range = [9.0e9, 11.0e9, 11]

[[ports]]
type = "rectangular"
tag = 2
width = {a}
height = {b}

[[ports]]
type = "rectangular"
tag = 3
width = {a}
height = {b}

[pec]
tags = [1]
"""

sim = rapidfem._native.Simulation.from_arrays(
    mesh.points, mesh.tets.astype(np.int64), tet_tags, tris, tri_tags, config
)
result = sim.run_sweep()
s = np.asarray(result.sparams)
s11_max = float(np.abs(s[:, 0, 0]).max())
s21 = np.abs(s[:, 1, 0])
print(f"max |S11| = {s11_max:.5f}  (gate < 0.01)")
print(f"|S21| range = [{s21.min():.5f}, {s21.max():.5f}]  (gate ~1 +- 0.01)")
ok = s11_max < 0.01 and abs(s21.min() - 1.0) < 0.01 and abs(s21.max() - 1.0) < 0.01
print("OK" if ok else "FAIL")
