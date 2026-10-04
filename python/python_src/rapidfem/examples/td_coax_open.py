"""Time-domain DGTD: a coax-fed monopole radiating into free space.

A 50 ohm air coax pokes in from one edge of an air box; its inner conductor
extends past the open end of the outer shield as a bare monopole. The
`rf.CoaxPort` feeds the TEM mode; the pulse travels down the line and the
protruding inner conductor radiates into the surrounding air, which is
terminated by ABCs so the field leaves without ringing. The 3-D animation
shows the TEM mode running down the coax and the wavefront launching off the
monopole.

Just a coax in an open box: the outer-conductor shield and the inner
conductor (including the protruding monopole) are PEC cylinder walls, and
the box edges are ABCs. No ground flange, no metal enclosure.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

mm = 1e-3
RI, RO = 1.50e-3, 3.45e-3   # inner / outer radius (50 ohm air coax)
LIN = 25.0e-3              # how far the outer-conductor coax pokes into the box
LPROT = 10.0e-3           # inner conductor protrudes past the shield (monopole)
BW = 44.0e-3               # air-box cross-section
BL = 70.0e-3              # air-box length (coax + free space ahead)
F0 = 8.0e9

# %% Geometry: air coax poking in, inner conductor extended into a monopole
Z0 = -BL / 2               # box -z face (the coax feed end / edge)
Z1 = Z0 + LIN             # open end of the outer-conductor shield
Z2 = Z1 + LPROT           # tip of the protruding inner-conductor monopole
# The free-space box at 6 cells per wavelength (order-2 DG carries the wave
# fine at that, and it is the bulk of the mesh); the coax keeps RO / 3.
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=10.0e9, per_lambda=6))
box = g.box(BW, BW, BL, position=(-BW / 2, -BW / 2, Z0), material=rf.Air())
coax = g.cylinder(radius=RO, height=LIN, position=(0, 0, Z0), axis=(0, 0, 1),
                  material=rf.Air(), maxh=RO / 3)
inner = g.cylinder(radius=RI, height=LIN + LPROT, position=(0, 0, Z0),
                   axis=(0, 0, 1), material=rf.Air(), maxh=RO / 3)
g.fragment(box, coax, inner)

p_in = rf.CoaxPort(coax.faces.min(axis="z"), ri=RI, ro=RO, origin=(0, 0, Z0))
# Outer-conductor shield: the coax side wall (centroid at its mid-length).
rf.PEC(*coax.faces.where(lambda c, b: Z0 + 1e-4 < c[2] < Z1 - 1e-4))
# Inner conductor + protruding monopole: the full inner side wall (Z0..Z2)
# plus the monopole tip cap, so it is a solid PEC rod radiating in the air.
rf.PEC(*inner.faces.where(lambda c, b: Z0 + 1e-4 < c[2] < Z2 - 1e-4))
rf.PEC(inner.faces.max(axis="z"))
rf.ABC(*box.faces.outer)
g.mesh()
rf.show(g)

# %% Build the time-domain problem
ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD coax monopole - {ptd.n_tets} tets, {ptd.n_dofs} state DOFs, "
      f"monopole {LPROT / mm:.0f} mm, TEM drive {F0 / 1e9:.1f} GHz")

# %% Drive the TEM mode and watch the pulse radiate from the open tip
# Exponential (Krylov ETD) propagator: the thin protruding monopole gives a
# few stiff mesh elements whose true CFL the explicit stepper's estimate
# overshoots (it diverges); the unconditionally stable exponential step has
# no such limit and the operator itself is well-behaved here.
pulse = rf.GaussianPulse(t0=60e-12, tau=16e-12, f0=F0)
traj = ptd.transient(
    port=p_in, waveform=pulse, dt=3e-12, steps=110,
    method="exponential", device="gpu",
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"open-coax transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: the TEM pulse launching off the protruding monopole
rf.show(traj)

# %% Port modal amplitude: incident pulse and the open-tip reflection
rf.show(ptd.port_signals(traj, [p_in], dt=3e-12, labels=["coax feed"]))
