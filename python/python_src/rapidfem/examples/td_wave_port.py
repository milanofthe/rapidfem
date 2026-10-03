"""Time-domain DGTD: a numerically-solved WavePort on a circular guide.

A circular waveguide has no closed-form Cartesian mode, so `rf.WavePort`
computes the transverse profile by a 2D eigensolve on the port-face
cross-section, then injects that solved mode. Here it launches the
fundamental TE11 mode of a hollow circular guide as a band-limited pulse;
it propagates and leaves through the matched output port. The 3-D
animation shows the solved TE11 field profile travelling down the guide.

This is the headline modal-port feature: the same `transient(port=...)`
injection as the analytic rect / coax ports, but the source pattern `b`
comes from a numerical cross-section eigensolve, so it works on guides
whose mode has no analytic form.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

R, L = 10.0e-3, 50.0e-3     # circular-guide radius / length [m]
F0 = 11.0e9                 # drive centre frequency (above TE11 cutoff)
MAXH = rf.lambda_maxh(f_max=13.0e9)

# %% Geometry: a hollow circular waveguide, WavePorts on the two ends
g = rf.Geometry(maxh=MAXH)
guide = g.cylinder(radius=R, height=L, position=(0, 0, 0), material=rf.Air())
p_in = rf.WavePort(guide.faces.min(axis="z"))    # numerical TE11 (scalar TE path, mode 0)
p_out = rf.WavePort(guide.faces.max(axis="z"))
rf.PEC(*guide.faces.unassigned)
g.mesh()
rf.show(g)

# %% Build the time-domain problem
ptd = rf.ProblemTD(g, order=2, flux="upwind")
# Circular-guide TE11 cutoff: f_c = 1.841 c / (2 pi R).
fc = 1.841 * ptd.c / (2.0 * np.pi * R)
print(f"DGTD circular guide - {ptd.n_tets} tets, {ptd.n_dofs} state DOFs, "
      f"TE11 cutoff {fc / 1e9:.2f} GHz, drive {F0 / 1e9:.1f} GHz")

# %% Drive the numerically-solved TE11 mode and watch it propagate
pulse = rf.GaussianPulse(t0=80e-12, tau=20e-12, f0=F0)
traj = ptd.transient(
    port=p_in, waveform=pulse, dt=2.5e-12, steps=280,
    method="explicit", device="gpu",
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"WavePort TE11 transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: the solved TE11 mode travelling down the circular guide
rf.show(traj)

# %% Port modal amplitudes over time (numerically-solved TE11 at both ports)
rf.show(ptd.port_signals(traj, [p_in, p_out], dt=2.5e-12,
                         labels=["input port", "output port"]))
