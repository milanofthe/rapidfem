"""WR-90 rectangular waveguide, 21-point sweep across the single-mode band.

Notebook-style flow:

    Parameters  ->  Geometry + Materials  ->  Mesh  ->  Problem

Each `# %%` block is a cell. Run them top-to-bottom with Shift+Enter, or hit
Run-All. The script also runs end-to-end as plain `python fd_wr90.py`, `# %%`
is just a comment.
"""

# %% Parameters
import numpy as np
import rapidfem as rf

A, B, L = 22.86e-3, 10.16e-3, 30.0e-3        # WR-90 width, height, length [m]
FREQUENCIES = np.linspace(8.0e9, 12.0e9, 21) # 21-point sweep, single-mode band
MAXH = rf.lambda_maxh(f_max=12.0e9)          # ~2.1 mm, air λ/12 at f_max


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())

rf.RectWaveguidePort(air.faces.min(axis="z"))
rf.RectWaveguidePort(air.faces.max(axis="z"))
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
print(f"|S11| at f0: {abs(result.sparams[0, 0, 0]):.4g}")
print(f"|S21| at f0: {abs(result.sparams[0, 1, 0]):.4g}")
