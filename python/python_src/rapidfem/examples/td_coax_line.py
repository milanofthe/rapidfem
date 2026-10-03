"""Time-domain DGTD: a TEM pulse on a 50 ohm coaxial line.

The input `rf.CoaxPort` injects the analytic coaxial TEM mode (the radial
`E_rho ~ rho_hat / rho` field between inner and outer conductor) as a
band-limited pulse. With no low-frequency cutoff the TEM pulse travels
non-dispersively down the line and leaves through the matched output port.
The 3-D animation shows the annular field pattern sweeping along the coax.

Modal-port injection via `transient(port=...)`: the coax mode is the source
pattern `b` in `dy/dt = A·y + b·g(t)`. The outer cylinder is air; the inner
conductor is carved out with `g.fragment`, so its surface and the outer wall
become PEC, leaving the two flat ends as the coaxial ports.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

RI, RO = 1.50e-3, 3.45e-3   # inner / outer radius, 50 ohm air coax [m]
L = 40.0e-3                 # line length [m]
F0 = 8.0e9                  # drive centre frequency

# %% Geometry: an air coax with the inner conductor fragmented out
g = rf.Geometry(maxh=RO / 3.0)
air = g.cylinder(radius=RO, height=L, position=(0, 0, 0), material=rf.Air())
inner = g.cylinder(radius=RI, height=L, position=(0, 0, 0), material=rf.Air())
g.fragment(air, inner)

p_in = rf.CoaxPort(air.faces.min(axis="z"), ri=RI, ro=RO, origin=(0, 0, 0))
p_out = rf.CoaxPort(air.faces.max(axis="z"), ri=RI, ro=RO, origin=(0, 0, L))
rf.PEC(*air.faces.unassigned)        # inner-conductor surface + outer wall
g.mesh()
rf.show(g)

# %% Build the time-domain problem
ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD 50 ohm coax - {ptd.n_tets} tets, {ptd.n_dofs} state DOFs, "
      f"TEM drive {F0 / 1e9:.1f} GHz")

# %% Drive the TEM mode at one end and watch the pulse travel
pulse = rf.GaussianPulse(t0=70e-12, tau=18e-12, f0=F0)
traj = ptd.transient(
    port=p_in, waveform=pulse, dt=2e-12, steps=150,
    method="explicit", device="gpu",
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"TEM transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: the coaxial TEM pulse running down the line
rf.show(traj)

# %% Port modal amplitudes over time (TEM wave at input and output)
rf.show(ptd.port_signals(traj, [p_in, p_out], dt=2e-12,
                         labels=["input port", "output port"]))
