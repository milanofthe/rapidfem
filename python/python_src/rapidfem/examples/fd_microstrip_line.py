"""50 ohm microstrip line on RO4003C, driven by hybrid wave ports.

A straight 30 mm microstrip section on 0.508 mm Rogers RO4003C
(er = 3.55). Both ends are excited by a full-vector wave port on the
substrate-plus-air cross-section, so the inhomogeneous quasi-TEM mode is
de-embedded directly and the line sees its proper modal reference
impedance at every frequency (no need to centre the sweep on a
travelling-wave resonance the way a lumped gap source would).

The 20 mm substrate width keeps the lowest transverse box resonance of
the cross-section (~4 GHz dielectric estimate, higher once the air region
pulls it up) above the 3.3 GHz top of the sweep, so the single-mode
wave-port projection stays energy-conserving across the band.
"""

# %% Parameters
import math
import numpy as np
import rapidfem as rf

mm = 1e-3

SUB_H  = 0.508 * mm
ER_SUB = 3.55
TAND   = 0.0027

LINE_W = 1.13 * mm
LINE_L = 30.0 * mm

SUB_W  = 20.0 * mm
AIR_H  = 10.0 * mm

FREQUENCIES = np.linspace(2.85e9, 3.30e9, 21)
MAXH = rf.lambda_maxh(f_max=3.3e9, er_max=ER_SUB)


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)

# The substrate volume carries the line at 5/3 of its thickness; only the
# wave-port cross-sections need ~1/3 of it (a 1-element-thick slab is too
# coarse for the vector eigensolve to see the quasi-TEM mode), so those faces
# are refined below. A mesh study: 15 times fewer tets than 1/3 everywhere,
# |S| within 0.002 of 2/3 everywhere. Air stays on the wavelength cap.
fr4 = rf.Dielectric(er=ER_SUB, tand=TAND, maxh=5 * SUB_H / 3)

sub = g.box(SUB_W, LINE_L, SUB_H, position=(-SUB_W / 2, 0, 0), material=fr4)
air = g.box(SUB_W, LINE_L, AIR_H, position=(-SUB_W / 2, 0, SUB_H),
            material=rf.Air())

trace = g.xy_plate(LINE_W, LINE_L, position=(-LINE_W / 2, 0, SUB_H))

g.fragment(sub, air, trace)

# Trace + ground plane: the cross-section's internal conductor plus its
# outer PEC. Both ride on one PEC object so the wave-port eigensolve can
# mark the right nodes via pec=[pec_strip].
pec_strip = rf.PEC(trace, sub.faces.min(axis="z"))

# Wave ports on the full microstrip cross-section at each end of the line
# (it runs along y, so the port faces are the substrate + air y-extremes).
# f0 at band centre fixes beta = n_eff(f0) * k0 for the sweep.
F0 = 0.5 * (FREQUENCIES[0] + FREQUENCIES[-1])
rf.WavePort(sub.faces.min(axis="y"), air.faces.min(axis="y"),
            f0=F0, mode_kind="auto", pec=[pec_strip])
rf.WavePort(sub.faces.max(axis="y"), air.faces.max(axis="y"),
            f0=F0, mode_kind="auto", pec=[pec_strip])

# Open the enclosure: first-order ABC on the lateral x-walls (substrate and
# air) and the air top. The y-extreme faces are the wave ports, so they are
# excluded here.
rf.ABC(sub.faces.min(axis="x"), sub.faces.max(axis="x"),
       air.faces.min(axis="x"), air.faces.max(axis="x"),
       air.faces.max(axis="z"))

rf.show(g)


# %% Mesh
for port_face in (sub.faces.min(axis="y"), sub.faces.max(axis="y")):
    port_face.maxh = SUB_H / 3
g.auto_refine_features(base_maxh=MAXH)
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.ProblemFD(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

s11_db = [20 * math.log10(max(abs(result.sparams[i, 0, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
s21_db = [20 * math.log10(max(abs(result.sparams[i, 1, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
print(f"|S11| across band: {min(s11_db):.1f} to {max(s11_db):.1f} dB")
print(f"|S21| across band: {min(s21_db):.2f} to {max(s21_db):.2f} dB")
