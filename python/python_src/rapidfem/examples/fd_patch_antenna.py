"""Edge-fed microstrip patch antenna on FR-4, single frequency, with far-field.

A ~2.4 GHz patch with a lumped feed at the radiating edge, a full PML
enclosure on four side walls + the +z cap, and a ground plane at the
bottom. After the solve, ``prob.farfield`` gives a directivity pattern.

The PML is built from five non-overlapping cuboid slabs (one per outer
face of the air box). The ±x slabs are extended in y to cover the
y-corners, and the +z slab is extended in both x and y to cover the top
corners. Every outgoing ray hits exactly one PML on its way out.
"""

# %% Parameters
import numpy as np
import rapidfem as rf

mm = 1e-3

SUB_W, SUB_L, SUB_H = 60 * mm, 60 * mm, 1.6 * mm
ER_SUB = 4.4

PATCH_W, PATCH_L = 38 * mm, 29 * mm

FEED_X, FEED_Y = 0.0, -PATCH_L / 2
FEED_WIDTH = 1.5 * mm

PAD_XY = 25 * mm
PAD_Z = 60 * mm
PML_T = 20 * mm

FREQUENCIES = np.linspace(2.0e9, 2.8e9, 21)
F0 = 2.4e9

MAXH = rf.lambda_maxh(f_max=2.8e9)


# %% Geometry + Materials
total_w = SUB_W + 2 * PAD_XY
total_l = SUB_L + 2 * PAD_XY
AIR_TOP = SUB_H + PAD_Z

X_OUT = total_w / 2
Y_OUT = total_l / 2

g = rf.Geometry(maxh=MAXH)

# Material-level sizing: one Dielectric for the substrate (FR-4),
# one Air for the bulk near-field region, and one PML-side Air at 2×
# coarser. The PML stretch makes a fine PML mesh wasteful, the
# absorber's accuracy is set by the polynomial profile, not the cell
# count, so we relax it.
fr4     = rf.Dielectric(er=ER_SUB, maxh=1.5 * SUB_H)
bulk_air = rf.Air()
pml_air  = rf.Air(maxh=2 * MAXH)

air = g.box(total_w, total_l, AIR_TOP, position=(-X_OUT, -Y_OUT, 0),
            material=bulk_air)

pml_xp = g.box(PML_T, total_l + 2 * PML_T, AIR_TOP,
               position=(X_OUT, -Y_OUT - PML_T, 0), material=pml_air)
pml_xm = g.box(PML_T, total_l + 2 * PML_T, AIR_TOP,
               position=(-X_OUT - PML_T, -Y_OUT - PML_T, 0), material=pml_air)
pml_yp = g.box(total_w, PML_T, AIR_TOP,
               position=(-X_OUT, Y_OUT, 0), material=pml_air)
pml_ym = g.box(total_w, PML_T, AIR_TOP,
               position=(-X_OUT, -Y_OUT - PML_T, 0), material=pml_air)
pml_zp = g.box(total_w + 2 * PML_T, total_l + 2 * PML_T, PML_T,
               position=(-X_OUT - PML_T, -Y_OUT - PML_T, AIR_TOP), material=pml_air)

sub = g.box(SUB_W, SUB_L, SUB_H, position=(-SUB_W / 2, -SUB_L / 2, 0),
            material=fr4)

patch = g.xy_plate(PATCH_W, PATCH_L,
                   position=(-PATCH_W / 2, -PATCH_L / 2, SUB_H))

feed = g.plate(
    p0=(FEED_X - FEED_WIDTH / 2, FEED_Y, 0),
    width=(FEED_WIDTH, 0, 0),
    height=(0, 0, SUB_H),
)

g.fragment(air, pml_xp, pml_xm, pml_yp, pml_ym, pml_zp, sub, patch, feed)

# Physics: feed port, PEC conductors, PML on each slab.
rf.LumpedPort(feed, direction=(0, 0, 1), z0=50.0)
rf.PEC(patch, sub.faces.min(axis="z"))   # patch + ground plane

rf.PML(pml_xp, direction=( 1, 0, 0), inner_face= X_OUT,   thickness=PML_T)
rf.PML(pml_xm, direction=(-1, 0, 0), inner_face=-X_OUT,   thickness=PML_T)
rf.PML(pml_yp, direction=( 0, 1, 0), inner_face= Y_OUT,   thickness=PML_T)
rf.PML(pml_ym, direction=( 0,-1, 0), inner_face=-Y_OUT,   thickness=PML_T)
rf.PML(pml_zp, direction=( 0, 0, 1), inner_face= AIR_TOP, thickness=PML_T)

# Outer hull of every PML slab → PEC (terminates the absorber).
rf.PEC(*pml_xp.faces.outer, *pml_xm.faces.outer,
       *pml_yp.faces.outer, *pml_ym.faces.outer,
       *pml_zp.faces.outer)

# Near-field-to-far-field surface. With a PML there is no ABC Huygens surface
# to auto-detect, so mark the bulk-air box boundary for the NFFT integral.
# .hull (not .outer): the air box is PML-wrapped on five sides, so only its
# z = 0 bottom face touches the model bbox; .outer would return that single
# face, not the closed box. .hull keys off air's own bbox and returns all six.
# Its z = 0 face lies on the ground plane: the solver takes that plane as an
# infinite ground and closes the surface with the images of the other five.
rf.FarFieldSurface(*air.faces.hull)

rf.show(g)


# %% Mesh
g.auto_refine_features(base_maxh=MAXH)
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.Problem(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")

mags = [abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))]
fi_min = int(min(range(len(mags)), key=lambda i: mags[i]))
print(f"|S11| min: {mags[fi_min]:.4f} @ {FREQUENCIES[fi_min]/1e9:.3f} GHz")

fi0 = int(min(range(len(FREQUENCIES)), key=lambda i: abs(FREQUENCIES[i] - F0)))
pattern = prob.farfield(result, freq_idx=fi0, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"Far-field @ {FREQUENCIES[fi0]/1e9:.2f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
