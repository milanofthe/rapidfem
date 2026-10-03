"""3-iris X-band bandpass filter in WR-90.

Three inductive irises (thin PEC strips spanning the y-direction) inside
the waveguide form a 3rd-order Chebyshev-style bandpass response around
10 GHz. S21 should drop sharply outside the passband.
"""

# %% Parameters
import math
import numpy as np
import rapidfem as rf

A, B = 22.86e-3, 10.16e-3

APERTURES = [10.0e-3, 8.0e-3, 10.0e-3]
SPACING   = 15.0e-3
IRIS_T    = 1.0e-3

INPUT_LEN  = 12.0e-3
OUTPUT_LEN = 12.0e-3
L = INPUT_LEN + (len(APERTURES) - 1) * SPACING + 2 * IRIS_T + OUTPUT_LEN

FREQUENCIES = np.linspace(8.2e9, 12.4e9, 41)
MAXH = rf.lambda_maxh(f_max=12.4e9)


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())

# Iris z-centers, from input toward output.
z_centers = [INPUT_LEN + IRIS_T / 2 + k * SPACING for k in range(len(APERTURES))]

# Each iris = two PEC strips along ±x leaving a slot in x of APERTURES[k].
# The iris-solid volumes are dummies, their faces become PEC via fragment,
# and the tets inside are just air (no field of interest behind a PEC wall).
iris_vols = []
for k, (zc, w) in enumerate(zip(z_centers, APERTURES)):
    slot = w
    strip_w = (A - slot) / 2
    for side in (-1, +1):
        x0 = -A / 2 if side < 0 else slot / 2
        iris = g.box(strip_w, B, IRIS_T, position=(x0, -B / 2, zc - IRIS_T / 2),
                     material=rf.Air())
        iris_vols.append(iris)

g.fragment(air, *iris_vols)

# Ports + PEC. air.faces after fragment includes the iris-interface faces,
# so .unassigned (after attaching ports) gives both waveguide walls AND
# all iris boundary faces, exactly what we want PEC'd.
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

mags = [abs(result.sparams[i, 1, 0]) for i in range(len(FREQUENCIES))]
db = [20 * math.log10(max(m, 1e-12)) for m in mags]
passband = [FREQUENCIES[i] / 1e9 for i, d in enumerate(db) if d > -3]
print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
if passband:
    print(f"3-dB band: {passband[0]:.2f}-{passband[-1]:.2f} GHz ({len(passband)} pts)")
else:
    print("No 3-dB passband found in the sweep, try widening the frequency range.")
