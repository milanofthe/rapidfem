"""Pyramidal horn antenna, WR-90 feed + PML in the main-beam direction.

A short WR-90 feed waveguide followed by an 80 mm taper to a 30×22 mm
aperture. The +x face is terminated by a PML slab; the other five outer
faces use a 1st-order ABC.
"""

# %% Parameters
import numpy as np
import rapidfem as rf

mm = 1e-3

wga, wgb = 22.86 * mm, 10.16 * mm
Lfeed = 15.0 * mm

Lhorn = 50.0 * mm
WH, HH = 30.0 * mm, 22.0 * mm

LPAD_BEAM = 88.0 * mm   # +x pad: roomy for the diverging beam before the PML
LPAD_SIDE = 30.0 * mm   # y/z pad: tighter, the side ABC just terminates near-field leakage
PML_T = 15.0 * mm

FREQUENCIES = np.linspace(8.0e9, 12.0e9, 15)
F0 = 10.0e9

MAXH = rf.lambda_maxh(f_max=11.0e9, per_lambda=8)        # horn / feed metals
MAXH_AIR = rf.lambda_maxh(f_max=11.0e9, per_lambda=3)    # outer air padding


# %% Geometry + Materials
AIR_X0, AIR_X1 = -Lfeed,             Lhorn + LPAD_BEAM
AIR_Y0, AIR_Y1 = -WH / 2 - LPAD_SIDE, WH / 2 + LPAD_SIDE
AIR_Z0, AIR_Z1 = -HH / 2 - LPAD_SIDE, HH / 2 + LPAD_SIDE

# Global cap is the *coarse* outer-air size; the horn flare and feed are
# refined locally below.
g = rf.Geometry(maxh=MAXH_AIR)

air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
            position=(AIR_X0, AIR_Y0, AIR_Z0),
            material=rf.Air())

pml_xp = g.box(PML_T, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
               position=(AIR_X1, AIR_Y0, AIR_Z0),
               material=rf.Air(), maxh=2 * MAXH)

feed = g.box(Lfeed, wga, wgb, position=(-Lfeed, -wga / 2, -wgb / 2),
             material=rf.Air(), maxh=wgb / 3)

throat = g.polygon([
    (0, -wga / 2, -wgb / 2), (0,  wga / 2, -wgb / 2),
    (0,  wga / 2,  wgb / 2), (0, -wga / 2,  wgb / 2),
])
aperture = g.polygon([
    (Lhorn, -WH / 2, -HH / 2), (Lhorn,  WH / 2, -HH / 2),
    (Lhorn,  WH / 2,  HH / 2), (Lhorn, -WH / 2,  HH / 2),
])
horn = g.loft(throat, aperture, material=rf.Air(), maxh=MAXH)

g.fragment(air, feed, horn, pml_xp)

# Physics
rf.RectWaveguidePort(feed.faces.min(axis="x"), mode=(1, 0), power=1.0)

# Feed waveguide PEC walls, four yz-side faces (skipping throat at x=0
# and port at x=-Lfeed).
rf.PEC(feed.faces.min(axis="y"), feed.faces.max(axis="y"),
       feed.faces.min(axis="z"), feed.faces.max(axis="z"))

# Four trapezoidal flares of the horn (excluding the throat and aperture caps).
rf.PEC(*horn.faces.where(lambda c, b: 1e-6 < c[0] < Lhorn - 1e-6))

rf.PML(pml_xp, direction=(1, 0, 0), inner_face=Lhorn + LPAD_BEAM,
       thickness=PML_T)

# PML outer hull faces → PEC (terminate the absorber).
rf.PEC(*pml_xp.faces.outer)

# Air outer hull (minus the PML interface, which is interior to the model) → ABC.
rf.ABC(*air.faces.outer.unassigned)

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
print(f"|S11| range: {min(abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))):.3f}, "
      f"{max(abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))):.3f}")

fi0 = int(min(range(len(FREQUENCIES)), key=lambda i: abs(FREQUENCIES[i] - F0)))
pattern = prob.farfield(result, freq_idx=fi0, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"Far-field @ {FREQUENCIES[fi0] / 1e9:.1f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
