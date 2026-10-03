"""Time-domain DGTD: a TE10 pulse reflecting off a dielectric slab.

A WR-90 guide carries an air section, a dielectric-filled middle section,
then another air section. A TE10 pulse launched at the input modal port
hits the air-to-dielectric interface, where the impedance step splits it:
part reflects back toward the input, part transmits through the slab and
on to the output port. The 3-D animation shows the incident pulse, the
reflection running backward, and the slowed-down transmitted pulse.

Both `rf.RectWaveguidePort`s sit in air-filled cross-sections, so the
analytic TE10 mode stays valid for injection and absorption; the slab is a
plain `rf.Dielectric` region the DGTD operator handles through its
per-element permittivity. A driven modal port meeting a material
discontinuity, the time-domain picture of a waveguide impedance step.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

A, B = 22.86e-3, 10.16e-3       # WR-90 cross-section [m]
L_AIR, L_SLAB = 24.0e-3, 16.0e-3  # air feed/exit length, dielectric slab length
ER = 2.5                        # slab relative permittivity
F0 = 10.0e9                     # drive centre frequency
MAXH = rf.lambda_maxh(f_max=12.0e9)

# %% Geometry: air | dielectric slab | air, stacked along z
g = rf.Geometry(maxh=MAXH)
air_in = g.box(A, B, L_AIR, position=(-A / 2, -B / 2, 0), material=rf.Air())
slab = g.box(A, B, L_SLAB, position=(-A / 2, -B / 2, L_AIR),
             material=rf.Dielectric(er=ER))
air_out = g.box(A, B, L_AIR, position=(-A / 2, -B / 2, L_AIR + L_SLAB),
                material=rf.Air())
g.fragment(air_in, slab, air_out)

p_in = rf.RectWaveguidePort(air_in.faces.min(axis="z"))
p_out = rf.RectWaveguidePort(air_out.faces.max(axis="z"))
# PEC only the four lateral walls of each section. The two air↔dielectric
# interfaces (the shared z-faces from fragment) must stay un-PEC'd: they
# are interior material boundaries the DG flux carries, so the pulse
# partially transmits into the slab instead of seeing a conducting wall.
for sec in (air_in, slab, air_out):
    rf.PEC(sec.faces.min(axis="x"), sec.faces.max(axis="x"),
           sec.faces.min(axis="y"), sec.faces.max(axis="y"))
g.mesh()
rf.show(g)

# %% Build the time-domain problem
ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD dielectric step - {ptd.n_tets} tets, {ptd.n_dofs} state DOFs, "
      f"slab er={ER}, drive {F0 / 1e9:.1f} GHz")

# %% Drive TE10 at the input and watch it split at the slab
pulse = rf.GaussianPulse(t0=90e-12, tau=22e-12, f0=F0)
traj = ptd.transient(
    port=p_in, waveform=pulse, dt=3e-12, steps=170,
    method="explicit", device="gpu",
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"dielectric-step transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: incident, reflected, and transmitted TE10 pulses
rf.show(traj)

# %% Port modal amplitudes: incident/reflected at input, transmitted at output
rf.show(ptd.port_signals(traj, [p_in, p_out], dt=3e-12,
                         labels=["input port", "output port"]))
